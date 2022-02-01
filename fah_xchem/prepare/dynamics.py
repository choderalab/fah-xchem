"""
Prepare all X-ray structures for FAH

# Projects

    13430 : apo Mpro monomer His41(0) Cys145(0)
    13431 : apo Mpro monomer His41(+) Cys145(-)
    13432 : holo Mpro monomer His41(0) Cys145(0)
    13433 : holo Mpro monomer His41(+) Cys145(-)
    13434 : apo Mpro dimer His41(0) Cys145(0)
    13435 : apo Mpro dimer His41(+) Cys145(-)
    13436 : holo Mpro dimer His41(0) Cys145(0)
    13437 : holo Mpro dimer His41(+) Cys145(-)

Each RUN corresponds to a different fragment structure

# Manifest

`../structures/metadata.csv` : master index of fragment IDs and RUNs
```
,crystal_name,RealCrystalName,smiles,new_smiles,alternate_name,site_name,pdb_entry
1,Mpro-1q2w-2020-04-Bonanno_0,Mpro-1q2w-2020-04-Bonanno,C[C@H](O)CC(C)(C)O,NA,NA,Mpro-SARS1,1Q2W
3,Mpro-1wof-2020-04-Yang_0,Mpro-1wof-2020-04-Yang,CCOC(O)CC[C@H](C[C@@H]1CCNC1O)N[C@H](O)[C@H](CC(C)C)NC(O)[C@@H](NC(O)[C@H](C)NC(O)C1CC(C)ON1)C(C)C,NA,NA,Mpro-SARS1,1WOF
4,Mpro-2a5i-2020-04-Lee_0,Mpro-2a5i-2020-04-Lee,CCOC(O)[C@@H](O)C[C@@H](O)N(CCC(N)O)NC(O)[C@H](CC1CCCCC1)N[C@H](O)[C@H](CC(C)C)N[C@H](O)OCC1CCCCC1,NA,NA,Mpro-SARS1,2A5I
...
```

First column is used to identify RUN:
* `RUN0` is skipped
* `RUN1` is Mpro-1q2w-2020-04-Bonanno_0
* `RUN2` is skipped
* `RUN3` is Mpro-1wof-2020-04-Yang_0
...

"""
import os
import time
from pathlib import Path
import bz2
from enum import Enum
import yaml
import logging

import numpy as np
from pydantic import BaseModel, Field
from simtk import unit, openmm
from openff.toolkit.topology import Molecule
from simtk.openmm import app
from openmmforcefields.generators import SystemGenerator
import mdtraj as md
from openmmtools import integrators
from rich.progress import track
from openeye import oechem


class SimulationParameters(BaseModel):
    protein_forcefield: str = "amber14/protein.ff14SB.xml"
    solvent_forcefield: str = "amber14/tip3p.xml"
    small_molecule_forcefield: str = "openff-1.3.0"
    water_model: str = "tip3p"
    solvent_padding = 10.0 * unit.angstrom
    ionic_strength = (
        70 * unit.millimolar
    )  # assay buffer: 20 mM HEPES pH 7.3, 1 mM TCEP, 50 mM NaCl, 0.01% Tween-20, 10% glycerol
    pressure = 1.0 * unit.atmospheres
    collision_rate = 1.0 / unit.picoseconds
    temperature = 300.0 * unit.kelvin
    timestep = 4.0 * unit.femtoseconds
    iterations = 1000  # 1 ns equilibration
    nsteps_per_iteration = 250
    nsteps_per_snapshot = 250000  # 1 ns
    nsnapshots_per_wu = 20  # number of snapshots per WU


class FAHProject(BaseModel):
    project: str = Field(..., description="Folding@Home project code")
    project_dir: Path = Field(
        ..., description="Path to directory for Folding@Home project"
    )
    simulation_parameters: SimulationParameters = Field(
        SimulationParameters(), description="Parameters for OpenMM simulations"
    )

    def _setup_fah_run(
        self,
        destination_path,
        protein_pdb_filename,
        oemol=None,
        cache=None,
        restrain_rmsd=False,
    ):
        """
        Prepare simulation.

        Parameters
        ----------
        destination_path : str
            The path to the RUN to be created
        protein_pdb_filename : str
            Path to protein PDB file
        oemol : openeye.oechem.OEMol, optional, default=None
            The molecule to parameterize, with SDData attached
            If None, don't include the small molecule
        restrain_rmsd : bool, optional, default=False
            If True, restrain RMSD during first equilibration phase
        """
        # Prepare phases
        system_xml_filename = os.path.join(destination_path, "system.xml.bz2")
        integrator_xml_filename = os.path.join(destination_path, "integrator.xml.bz2")
        state_xml_filename = os.path.join(destination_path, "state.xml.bz2")

        # Check if we can skip setup
        openmm_files_exist = (
            os.path.exists(system_xml_filename)
            and os.path.exists(state_xml_filename)
            and os.path.exists(integrator_xml_filename)
        )
        if openmm_files_exist:
            return

        # Create barostat
        barostat = openmm.MonteCarloBarostat(
            self.simulation_parameters.pressure, self.simulation_parameters.temperature
        )

        # Create RUN directory if it does not yet exist
        os.makedirs(destination_path, exist_ok=True)

        # Load any molecule(s)
        molecule = None
        molecules = []
        if oemol is not None:
            molecule = Molecule.from_openeye(oemol, allow_undefined_stereo=True)
            molecule.name = "MOL"  # Ensure residue is MOL
            logging.info([res for res in molecule.to_topology().to_openmm().residues()])
            molecules = [molecule]

        # Create SystemGenerator
        forcefield_kwargs = {
            "removeCMMotion": False,
            "hydrogenMass": 3.0 * unit.amu,
            "constraints": app.HBonds,
            "rigidWater": True,
        }
        periodic_kwargs = {"nonbondedMethod": app.PME, "ewaldErrorTolerance": 2.5e-04}
        forcefields = [
            self.simulation_parameters.protein_forcefield,
            self.simulation_parameters.solvent_forcefield,
        ]
        openmm_system_generator = SystemGenerator(
            forcefields=forcefields,
            molecules=molecules,
            small_molecule_forcefield=self.simulation_parameters.small_molecule_forcefield,
            cache=cache,
            barostat=barostat,
            forcefield_kwargs=forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_kwargs,
        )

        # Read protein
        logging.info(f"Reading protein from {protein_pdb_filename}...")
        pdbfile = app.PDBFile(protein_pdb_filename)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)

        if oemol is not None:
            # Add small molecule to the system
            modeller.add(molecule.to_topology().to_openmm(), molecule.conformers[0])

        # Extract protein and molecule chains and indices before adding solvent
        mdtop = md.Topology.from_openmm(modeller.topology)  # excludes solvent and ions
        protein_atom_indices = mdtop.select("protein and (mass > 1)")
        molecule_atom_indices = mdtop.select(
            "(not protein) and (not water) and (mass > 1)"
        )
        protein_chainids = list(
            set(
                [
                    atom.residue.chain.index
                    for atom in mdtop.atoms
                    if atom.index in protein_atom_indices
                ]
            )
        )
        n_protein_chains = len(protein_chainids)
        protein_chain_atom_indices = dict()
        for chainid in protein_chainids:
            protein_chain_atom_indices[chainid] = mdtop.select(
                f"protein and chainid {chainid}"
            )

        # Add solvent
        logging.info("Adding solvent...")
        kwargs = {"padding": self.simulation_parameters.solvent_padding}
        modeller.addSolvent(
            openmm_system_generator.forcefield,
            model=self.simulation_parameters.water_model,
            ionicStrength=self.simulation_parameters.ionic_strength,
            **kwargs,
        )

        # Write initial model and select atom subsets and chains
        with bz2.open(
            os.path.join(destination_path, "initial-model.pdb.bz2"), "wt"
        ) as outfile:
            app.PDBFile.writeFile(
                modeller.topology, modeller.positions, outfile, keepIds=True
            )

        # Create an OpenMM system
        logging.info("Creating OpenMM system...")
        system = openmm_system_generator.create_system(modeller.topology)

        #
        # Add virtual bonds to ensure protein subunits and ligand are imaged together
        #

        virtual_bond_force = openmm.CustomBondForce("0")
        system.addForce(virtual_bond_force)

        # Add a virtual bond between protein chains
        if n_protein_chains > 1:
            chainid = protein_chainids[0]
            iatom = protein_chain_atom_indices[chainid][0]
            for chainid in protein_chainids[1:]:
                jatom = protein_chain_atom_indices[chainid][0]
                logging.info(f"Creating virtual bond between atoms {iatom} and {jatom}")
                virtual_bond_force.addBond(int(iatom), int(jatom), [])

        # Add a virtual bond between protein and ligand to make sure they are not imaged separately
        if oemol is not None:
            ligand_atom_indices = mdtop.select(
                "((resname MOL) and (mass > 1))"
            )  # ligand heavy atoms
            protein_atom_index = int(protein_atom_indices[0])
            ligand_atom_index = int(ligand_atom_indices[0])
            logging.info(
                f"Creating virtual bond between atoms {protein_atom_index} and {ligand_atom_index}"
            )
            virtual_bond_force.addBond(
                int(protein_atom_index), int(ligand_atom_index), []
            )

        # Add RMSD restraints if requested
        if restrain_rmsd:
            logging.info("Adding RMSD restraint...")
            kB = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB
            kT = kB * self.simulation_parameters.temperature
            rmsd_atom_indices = mdtop.select(
                "(protein and (name CA)) or ((resname MOL) and (mass > 1))"
            )  # CA atoms and ligand heavy atoms
            rmsd_atom_indices = [int(index) for index in rmsd_atom_indices]
            custom_cv_force = openmm.CustomCVForce("(K_RMSD/2)*RMSD^2")
            custom_cv_force.addGlobalParameter("K_RMSD", kT / unit.angstrom**2)
            rmsd_force = openmm.RMSDForce(modeller.positions, rmsd_atom_indices)
            custom_cv_force.addCollectiveVariable("RMSD", rmsd_force)
            force_index = system.addForce(custom_cv_force)

        # Create OpenM Context
        platform = openmm.Platform.getPlatformByName("CPU")
        # platform = openmm.Platform.findPlatform()
        # platform.setPropertyDefaultValue('Precision', 'mixed')
        integrator = integrators.LangevinIntegrator(
            self.simulation_parameters.temperature,
            self.simulation_parameters.collision_rate,
            self.simulation_parameters.timestep,
        )
        context = openmm.Context(system, integrator, platform)
        context.setPositions(modeller.positions)

        # Report initial potential energy
        state = context.getState(getEnergy=True)
        logging.info(
            f"Initial potential energy is {state.getPotentialEnergy() / unit.kilocalories_per_mole:.3f} kcal/mol"
        )

        # Store snapshots in MDTraj trajectory to examine RMSD
        mdtop = md.Topology.from_openmm(pdbfile.topology)
        atom_indices = mdtop.select("all")  # all solute atoms
        protein_atom_indices = mdtop.select(
            "protein and (mass > 1)"
        )  # heavy solute atoms
        if oemol is not None:
            ligand_atom_indices = mdtop.select(
                "(resname MOL) and (mass > 1)"
            )  # ligand heavy atoms
        trajectory = md.Trajectory(
            np.zeros(
                [self.simulation_parameters.iterations + 1, len(atom_indices), 3],
                np.float32,
            ),
            mdtop,
        )
        trajectory.xyz[0, :, :] = (
            context.getState(getPositions=True).getPositions(asNumpy=True)[atom_indices]
            / unit.nanometers
        )

        # Minimize
        logging.info("Minimizing...")
        openmm.LocalEnergyMinimizer.minimize(context)

        # Equilibrate (with RMSD restraint if needed)
        initial_time = time.time()
        for iteration in track(
            range(self.simulation_parameters.iterations), "Equilibrating..."
        ):
            integrator.step(self.simulation_parameters.nsteps_per_iteration)
            trajectory.xyz[iteration + 1, :, :] = (
                context.getState(getPositions=True).getPositions(asNumpy=True)[
                    atom_indices
                ]
                / unit.nanometers
            )
        elapsed_time = (time.time() - initial_time) * unit.seconds
        ns_per_day = (context.getState().getTime() / elapsed_time) / (
            unit.nanoseconds / unit.day
        )
        logging.info(f"Performance: {ns_per_day:8.3f} ns/day")

        if restrain_rmsd:
            # Disable RMSD restraint
            context.setParameter("K_RMSD", 0.0)

            logging.info("Minimizing...")
            openmm.LocalEnergyMinimizer.minimize(context)

            for iteration in track(
                range(self.simulation_parameters.iterations),
                "Equilibrating without RMSD restraint...",
            ):
                integrator.step(self.simulation_parameters.nsteps_per_iteration)

        # Retrieve state
        state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, getForces=True
        )
        system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        modeller.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        logging.info(
            f"Final potential energy is {state.getPotentialEnergy() / unit.kilocalories_per_mole:.3f} kcal/mol"
        )

        # Equilibrate again if we restrained the RMSD
        if restrain_rmsd:
            logging.info("Removing RMSD restraint from system...")
            system.removeForce(force_index)

        # Save as OpenMM
        logging.info("Exporting for OpenMM FAH simulation...")
        with bz2.open(integrator_xml_filename, "wt") as f:
            f.write(openmm.XmlSerializer.serialize(integrator))
        with bz2.open(state_xml_filename, "wt") as f:
            f.write(openmm.XmlSerializer.serialize(state))
        with bz2.open(system_xml_filename, "wt") as f:
            f.write(openmm.XmlSerializer.serialize(system))
        with bz2.open(
            os.path.join(destination_path, "equilibrated-all.pdb.bz2"), "wt"
        ) as f:
            app.PDBFile.writeFile(
                modeller.topology, state.getPositions(), f, keepIds=True
            )
        with open(os.path.join(destination_path, "equilibrated-solute.pdb"), "wt") as f:
            mdtraj_topology = md.Topology.from_openmm(modeller.topology)
            mdtraj_trajectory = md.Trajectory(
                [state.getPositions(asNumpy=True) / unit.nanometers], mdtraj_topology
            )
            selection = mdtraj_topology.select("not water")
            mdtraj_trajectory = mdtraj_trajectory.atom_slice(selection)
            app.PDBFile.writeFile(
                mdtraj_trajectory.topology.to_openmm(),
                mdtraj_trajectory.openmm_positions(0),
                f,
                keepIds=True,
            )
        with open(os.path.join(destination_path, "core.xml"), "wt") as f:
            f.write(f"<config>\n")
            f.write(
                f"  <numSteps>{self.simulation_parameters.nsteps_per_snapshot * self.simulation_parameters.nsnapshots_per_wu}</numSteps>\n"
            )
            f.write(
                f"  <xtcFreq>{self.simulation_parameters.nsteps_per_snapshot}</xtcFreq>\n"
            )
            f.write(f"  <precision>mixed</precision>\n")
            f.write(
                f'  <xtcAtoms>{",".join([str(index) for index in selection])}</xtcAtoms>\n'
            )
            f.write(f"</config>\n")
        if oemol is not None:
            # Write molecule as SDF, SMILES, and mol2
            for extension in ["sdf", "mol2", "smi", "csv"]:
                filename = os.path.join(destination_path, f"molecule.{extension}")
                with oechem.oemolostream(filename) as ofs:
                    oechem.OEWriteMolecule(ofs, oemol)

        # Clean up
        del context, integrator

    def _get_bound_ligand(self, ligand_mol2, metadata):
        # Read molecule in the appropriate protonation state
        try:
            # TODO: deal with hardcodes in a more elegant way, if possible
            oemol = oechem.OEMol()

            with oechem.oemolistream(ligand_mol2) as ifs:
                oechem.OEReadMolecule(ifs, oemol)

            # Rename the molecule
            title = metadata["alternate_name"]
            logging.info(f"Setting title to {title}")
            oemol.SetTitle(title)

            # Remove dummy atoms
            for atom in oemol.GetAtoms():
                if atom.GetName().startswith("Du"):
                    logging.info("Removing dummy atom.")
                    oemol.DeleteAtom(atom)

            # Attach all structure metadata to the molecule
            for key in metadata:
                oechem.OESetSDData(oemol, key, metadata[key])
        except Exception as e:
            logging.info(e)
            oemol = None

        return oemol

    def generate_run(self, run, target_pdb, ligand_mol2, metadata, cache=None):
        destination_path = self.project_dir.joinpath("RUNS", f"RUN{run}")

        oemol = self._get_bound_ligand(ligand_mol2, metadata)

        # Create RUN directory if it does not yet exist
        os.makedirs(destination_path, exist_ok=True)

        # Write metadata
        with open(os.path.join(destination_path, "metadata.yaml"), "wt") as outfile:
            yaml.dump(dict(metadata), outfile)

        # Set up RUN
        self._setup_fah_run(destination_path, target_pdb, oemol=oemol, cache=cache)
