"""
Tools for extracting snapshots and structures from core22 FAH trajectories.

Limitations:
* The reference structure (`natoms_reference`) must share the same atom ordering as the first `natoms_reference` atoms of the trajectory.
  For now, this means that the SpruceTK prepared structure (`Mpro-x10789_0_bound-protein-thiolate.pdb`) is used

Dependencies:
* mdtraj >= 1.9.4 (conda-forge)

"""

from functools import partial
import logging
import pathlib
import multiprocessing
import os
import tempfile
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
import joblib
import mdtraj as md
import numpy as np

from ..schema import TransformationAnalysis, AnalysisConfig


class SnapshotArtifactory(BaseModel):
    """Structural snapshot creator."""

    config: AnalysisConfig
    project_dir: pathlib.Path = Field(
        description="Path to project directory (e.g. '/home/server/server2/projects/13422')"
    )
    project_data_dir: pathlib.Path = Field(
        description="Path to project data directory (e.g. '/home/server/server2/data/SVR314342810/PROJ13422')"
    )
    cache_dir: pathlib.Path = Field(
        None,
        description="If specified, cache relevant parts of 'htf.npz' file in a local directory of this name",
    )

    @staticmethod
    def _transformation_to_file_mapping(output_dir, run_id, ligand):
        fnames = [
            f"{ligand}_protein.pdb",
            f"{ligand}_complex.pdb",
            f"{ligand}_ligand.sdf",
        ]

        outfiles = [
            os.path.join(output_dir, f"RUN{run_id}", f"{fname}") for fname in fnames
        ]

        return outfiles

    @staticmethod
    def regenerate_atom_mappings(
        project_dir: str,
        run: int,
        hybrid_atom_mappings_path: str,
    ):
        """
        Regenerate the RUN hybrid_atom_mappings.npz file if key atom mappings are missing.

        NOTE: This is very slow, and the need for this will (hopefully) be eliminated by
        https://github.com/choderalab/perses/issues/908

        Parameters
        ----------
        project_dir : str
            Path to project directory (e.g. '/home/server/server2/projects/13422')
        run : int
            Run (e.g. 0)
        hybrid_atom_mappings_path : str
            Path to hybrid atom mappings npz file to generate
        """
        # Load hybrid topology factory
        htf_path = os.path.join(project_dir, "RUNS", f"RUN{run}", "htf.npz")
        if not os.path.exists(htf_path):
            logging.warning(
                f"{pdbfile_path} does not exist. {htf_path} not found, so unable to regenerate hybrid_atom_mappings.npz"
            )
            raise ValueError(f"Failed to load PDB file: {e}")

        logging.warning(f"Regenerating {hybrid_atom_mappings_path} from {htf_path}")
        # TODO: This is very fragile because it requres *exactly* the same versions of tools that generated the pickle to be installed
        # Replace this when able
        import openmm  # openmm 7.6 or later, needed for perses htf import

        # from openeye import oechem # needed for perses htf import
        htf = np.load(htf_path, allow_pickle=True)["arr_0"].tolist()

        # Determine atom indices stored by FAH core
        # This replicates the code in https://github.com/FoldingAtHome/openmm-core/blob/core22/src/OpenMMCore.cpp#L1435-L1464
        import openmm
        from openmm import unit

        system = htf.hybrid_system
        platform = openmm.Platform.getPlatformByName("Reference")
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, integrator, platform)
        molecules = context.getMolecules()
        del context, integrator
        non_water_atoms = list()
        for molecule in molecules:
            particles = 0
            total_mass = 0.0
            for atom_index in molecule:
                mass = system.getParticleMass(atom_index) / unit.amu
                total_mass += mass
                if mass > 0.0:
                    particles += 1
            if (particles != 3) or (total_mass >= 20.0):
                for atom_index in molecule:
                    non_water_atoms.append(int(atom_index))
        # hybrid_to_xtc_map[hybrid_atom_index] is the atom index in the XTC for hybrid_atom_index in the full hybrid System, if present
        hybrid_to_xtc_map = {
            non_water_atoms[xtc_atom_index]: xtc_atom_index
            for xtc_atom_index in range(len(non_water_atoms))
        }
        # xtc_to_hybrid_map[xtc_atom_index] is the atom index in the full hybrid System corrsponding to XTC atom index xtc_atom_index
        xtc_to_hybrid_map = {v: k for (k, v) in hybrid_to_xtc_map.items()}

        # Determine atom indices in the non-water old and new PDB files
        # Reproduce code added in perses in https://github.com/choderalab/perses/pull/839/files#diff-3c5caedfcf63266b94c0f07d1c15a050900b3d4c2aac336ec0a4f3d38290a2d3L416
        # and MDTraj remove_solvent: https://github.com/mdtraj/mdtraj/blob/62269309ef3b3c465bfc4f76bdcdf9522f5b2d16/mdtraj/core/trajectory.py#L1826-L1859
        def get_solute_indices(topology, exclude=["CL", "NA"]):
            from mdtraj.core.residue_names import _SOLVENT_TYPES

            solvent_types = list(_SOLVENT_TYPES)
            for solvent_type in exclude:
                if solvent_type not in solvent_types:
                    raise ValueError(solvent_type + "is not a valid solvent type")
                solvent_types.remove(solvent_type)
            atom_indices = [
                atom.index
                for atom in topology.atoms
                if atom.residue.name not in solvent_types
            ]
            return atom_indices

        old_topology = md.Topology.from_openmm(htf._topology_proposal.old_topology)
        old_solute_indices = get_solute_indices(old_topology)
        hybrid_to_old_solute_map = {
            htf._old_to_hybrid_map[
                old_solute_indices[old_solute_index]
            ]: old_solute_index
            for old_solute_index in range(len(old_solute_indices))
        }
        old_solute_to_xtc_map = {
            old_solute_index: hybrid_to_xtc_map[hybrid_index]
            for (hybrid_index, old_solute_index) in hybrid_to_old_solute_map.items()
        }

        new_topology = md.Topology.from_openmm(htf._topology_proposal.new_topology)
        new_solute_indices = get_solute_indices(new_topology)
        hybrid_to_new_solute_map = {
            htf._new_to_hybrid_map[
                new_solute_indices[new_solute_index]
            ]: new_solute_index
            for new_solute_index in range(len(new_solute_indices))
        }
        new_solute_to_xtc_map = {
            new_solute_index: hybrid_to_xtc_map[hybrid_index]
            for (hybrid_index, new_solute_index) in hybrid_to_new_solute_map.items()
        }

        # Save atom mappings, following format in
        # https://github.com/choderalab/perses/pull/839/files#diff-3c5caedfcf63266b94c0f07d1c15a050900b3d4c2aac336ec0a4f3d38290a2d3R430-R438
        np.savez(
            hybrid_atom_mappings_path,
            hybrid_to_old_map=htf._hybrid_to_old_map,
            hybrid_to_new_map=htf._hybrid_to_new_map,
            old_nowater_to_hybrid_nowater_map=old_solute_to_xtc_map,
            new_nowater_to_hybrid_nowater_map=new_solute_to_xtc_map,
        )

    @staticmethod
    def load_trajectory(
        project_dir: str,
        project_data_dir: str,
        ligand: str,
        run: int,
        clone: int,
        gen: int,
    ) -> md.Trajectory:
        """
        Load the trajectory from the specified PRCG.

        Parameters
        ----------
        project_dir : str
            Path to project directory (e.g. '/home/server/server2/projects/13422')
        project_data_dir : str
            Path to project data directory (e.g. '/home/server/server2/data/SVR314342810/PROJ13422')
        ligand : str
            Ligand topology to use: one of ['old', 'new']
        run : int
            Run (e.g. 0)
        clone : int
            Clone (e.g. 0)
        gen : int
            Gen (e.g. 0)

        Returns
        -------
        trajectory : mdtraj.Trajectory
          The trajectory

        """
        # Sanity checks
        if not ligand in ["old", "new"]:
            raise ValueError(
                f"ligand must be one of ['old', 'new']; instead got {ligand}"
            )

        # Load Topology for old or new ligand complex
        pdbfile_path = os.path.join(
            project_dir, "RUNS", f"RUN{run}", f"{ligand}_complex.pdb"
        )
        try:
            pdbfile = md.load(pdbfile_path)
        except OSError as e:
            raise ValueError(f"Failed to load PDB file: {e}")
        topology = pdbfile.topology
        # Slice out counterions
        # TODO: Figure out why counterions are in the PDB file to begin with; that shouldn't be happening!
        def get_solute_indices(topology, exclude=[]):
            from mdtraj.core.residue_names import _SOLVENT_TYPES

            solvent_types = list(_SOLVENT_TYPES)
            for solvent_type in exclude:
                if solvent_type not in solvent_types:
                    raise ValueError(solvent_type + "is not a valid solvent type")
                solvent_types.remove(solvent_type)
            atom_indices = [
                atom.index
                for atom in topology.atoms
                if atom.residue.name not in solvent_types
            ]
            return atom_indices

        n_topology_atoms_1 = topology.n_atoms  # DEBUG
        topology = topology.subset(get_solute_indices(topology))
        n_topology_atoms_2 = topology.n_atoms  # DEBUG

        # Attempt to slice out real atoms
        hybrid_atom_mappings_path = os.path.join(
            project_dir, "RUNS", f"RUN{run}", "hybrid_atom_mappings.npz"
        )
        mappings = np.load(hybrid_atom_mappings_path, allow_pickle=True)
        if "hybrid_solute_atom_indices" in mappings:
            # New-style mappings
            hybrid_atom_indices = [
                int(index) for index in mappings[f"hybrid_solute_atom_indices"]
            ]  # solute hybrid atom indices stored in the XTC (and hence PDB)
            n_hybrid_solute = len(hybrid_atom_indices)
            hybrid_to_real_map = mappings[f"hybrid_to_{ligand}_map"].flat[
                0
            ]  # mapping from full hybrid indices to full real indices
            real_to_hybrid_map = {int(v): int(k) for k, v in hybrid_to_real_map.items()}
            real_indices_in_pdb = np.sort(
                [int(index) for index in hybrid_to_real_map.values()]
            )  # indices of real system that appear in solute-only PDB, in correct order
            hybrid_atom_indices = [
                hybrid_atom_indices.index(real_to_hybrid_map[real_index])
                for real_index in real_indices_in_pdb
                if real_to_hybrid_map[real_index] in hybrid_atom_indices
            ]
            logging.info(
                f"{pdbfile_path} : hybrid solute has {n_hybrid_solute}; topology had : {n_topology_atoms_1} -> {n_topology_atoms_2}; {ligand} solute slice has {len(hybrid_atom_indices)} atoms"
            )
        else:
            # Old-style mappings: Need to use hybrid topology file
            # Load atom index mappings to slice the hybrid ligand and protein/ions out of the hybrid system
            hybrid_atom_mappings_path = os.path.join(
                project_dir, "RUNS", f"RUN{run}", "hybrid_atom_mappings-new.npz"
            )
            if not os.path.exists(hybrid_atom_mappings_path):
                SnapshotArtifactory.regenerate_atom_mappings(
                    project_dir, run, hybrid_atom_mappings_path
                )

            mappings = np.load(hybrid_atom_mappings_path, allow_pickle=True)
            real_to_hybrid_atom_map = mappings[
                f"{ligand}_nowater_to_hybrid_nowater_map"
            ].tolist()
            hybrid_atom_indices = [
                int(real_to_hybrid_atom_map[real_atom])
                for real_atom in range(len(real_to_hybrid_atom_map))
            ]

        # Load the hybrid xtc trajectory
        # TODO: Reuse path logic from fah_xchem.lib
        trajectory_path = os.path.join(
            project_data_dir,
            f"RUN{run}",
            f"CLONE{clone}",
            f"results{gen}",
            "positions.xtc",
        )
        try:
            from mdtraj.formats import XTCTrajectoryFile

            with XTCTrajectoryFile(trajectory_path) as xtcfile:
                xyz, time, step, box = xtcfile.read(atom_indices=hybrid_atom_indices)
        except OSError as e:
            raise ValueError(f"Failed to load trajectory: {e}")

        # Create the Trajectory for the real system
        trajectory = md.Trajectory(xyz, topology)

        return trajectory

    @staticmethod
    def load_fragment(
        structure_path: pathlib.Path,
        target_name: str,
        fragment_id: str,
        annotations: str,
        component: str,
    ) -> md.Trajectory:
        """
        Load the reference fragment structure

        Parameters
        ----------
        structure_path : pathlib.Path
            Path to reference structure directory.
        target_name : str
            Name of target (e.g. 'Mpro').
        fragment_id : str
            Fragment ID (e.g. 'x10789').
        annotations : str
            Additional characters in the reference file name (e.g. '_0A_bound').
        component : str
            Component of the system the reference corresponds to (e.g. 'protein')

        Returns
        -------
        fragment : mdtraj.Trajectory
            The fragment structure

        """
        # several components here: path, target name, fragment id, annotations (e.g. "0A_bound"), and component (e.g. "protein", "ligand")
        # separated by hyphens

        # TODO: Put this in the covid-moonshot path, or generalize to an arbitrary file
        # fragment = md.load(
        #    f"/home/server/server2/projects/available/covid-moonshot/receptors/monomer/Mpro-{fragment_id}_0A_bound-protein.pdb"
        # )
        fragment = md.load(
            f"{structure_path}/{target_name}-{fragment_id}{annotations}_bound-{component}.pdb"
        )

        return fragment

    def _mdtraj_to_oemol(self, snapshot: md.Trajectory):
        """
        Create an OEMol from an MDTraj file by writing and reading

        NOTE: This uses terrible heuristics

        Parameters
        ----------
        snapshot : mdtraj.Trajectory
            MDTraj Trajectory with a single snapshot

        Returns
        -------
        oemol : openeye.oechem.OEMol
            The OEMol

        """
        from openeye import oechem

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "tmp.pdb")
            # Write the PDB file
            snapshot.save(filename)
            # Read it with OpenEye
            with oechem.oemolistream(filename) as ifs:
                for mol in ifs.GetOEGraphMols():
                    return mol

    def extract_snapshot(
        self,
        project_dir: str,
        project_data_dir: str,
        ligand: str,
        run: int,
        clone: int,
        gen: int,
        frame: int,
        fragment_id: str,
    ):
        """
        Extract the specified snapshot, align it to the reference fragment, and write protein and ligands to separate PDB files

        Parameters
        ----------
        project_dir : str
            Path to project directory (e.g. '/home/server/server2/projects/13422')
        project_data_dir : str
            Path to project data directory
        ligand : str
            Which ligand topology to select: 'old' or 'new'
        run : str or int
            Run (e.g. '0')
        clone : str or int
            Clone (e.g. '0')
        gen : str or int
            Gen (e.g. '0')
        frame : int
        fragment_id : str
            Fragment ID (e.g. 'x10789')

        Returns
        -------
        sliced_snapshot : dict of str : mdtraj.Trajectory
          sliced_snapshot[name] is the Trajectory for name in ['protein', 'ligand', 'complex'] for the specified ligand ['old', 'new']
        components : dict of str : oechem.OEMol
          components[name] is the OEMol for name in ['protein', 'ligand'] for the specified ligand ['old', 'new']

        """
        # Load the trajectory
        trajectory = self.load_trajectory(
            project_dir, project_data_dir, ligand, run, clone, gen
        )

        # Load the fragment
        fragment = self.load_fragment(
            structure_path=self.config.structure_path,
            target_name=self.config.target_name,
            fragment_id=fragment_id,
            annotations=self.config.annotations,
            component=self.config.component,
        )

        # Align the trajectory to the fragment (in place)
        # TODO: Perform this imaging if the protein monomers and ligand are not connected with virtual bonds.
        # trajectory.image_molecules(inplace=True) # No need to image molecules anymore now that perses adds zero-energy bonds between protein and ligand!
        # trajectory.superpose(fragment, atom_indices=fragment.top.select("name CA"))

        # Superimpose using atom selection for active site
        # TODO: fix this hardcode for *MPro*!
        # We may not need to use a DSL at all if we can just select protein atoms
        atom_selection_dsl = "(name CA) and (residue 145 or residue 41 or residue 164 or residue 165 or residue 142 or residue 163)"
        trajectory.superpose(
            fragment,
            atom_indices=fragment.top.select(atom_selection_dsl),
        )  # DEBUG : Mpro active site only

        # Extract the snapshot
        snapshot = trajectory[frame]

        # Slice out protein, ligand, and complex components into a dict
        # This produces sliced_snapshot[component] for component in ['protein', 'ligand', 'complex']
        sliced_snapshot = self.slice_snapshot(
            snapshot, project_dir, run, self.cache_dir
        )

        # Convert to OEMol
        # NOTE: This uses heuristics, and should be replaced once we start storing actual chemical information
        # FIXME: This is producing incorrect new_ligand molecules, presumably because atom identities are incorrect
        components = dict()
        for name in ["protein", "ligand"]:
            components[name] = self._mdtraj_to_oemol(sliced_snapshot[name])

        return sliced_snapshot, components

    def slice_snapshot(
        self,
        snapshot: md.Trajectory,
        project_dir: str,
        run: int,
        cache_dir: Optional[str],
    ) -> Dict[str, md.Trajectory]:
        """
        Slice snapshot to specified state in-place

        Parameters
        ----------
        snapshot : mdtraj.Trajectory
           Snapshot to slice
        project_dir : str
           Path to project directory (e.g. '/home/server/server2/projects/13422')
        run : int
           Run (e.g. '0')
        cache_dir : str or None
           If specified, cache relevant parts of "htf.npz" file in a local directory of this name

        Returns
        -------
        sliced_snapshot : dict of str : mdtraj.Trajectory
          sliced_snapshot[x] where x is one of ['protein', 'ligand', 'complex']

        """
        # Define atom indices
        atom_indices = {
            "complex": snapshot.topology.select(
                "not water"
            ),  # <xtcAtoms v="solute"/> eliminates waters
            "protein": snapshot.topology.select(
                "protein and (mass > 1.1)"
            ),  # omit hydrogens for fragalysis
            "ligand": snapshot.topology.select(
                "resn MOL and (mass > 1.1)"
            ),  # omit hydrogens for fragalysis
        }

        sliced_snapshot = dict()
        for key, atom_indices in atom_indices.items():
            sliced_snapshot[key] = md.Trajectory(
                snapshot.xyz[:, atom_indices, :], snapshot.topology.subset(atom_indices)
            )

        return sliced_snapshot

    def generate_representative_snapshot(
        self,
        transformation: TransformationAnalysis,
        output_dir: str,
        overwrite: bool = False,
    ) -> None:
        r"""
        Generate representative snapshots for old and new ligands.

        Illustration of frames for nonequilibrium switching between states:

        old ---[0]\             /[3]
                   \           /
        new         \[1]---[2]/

        TODO: Can we use a more general scheme for extracting representative snapshots or snapshot ensembles?

        Parameters
        ----------
        transformation: TransformationAnalysis
            The transformation record to operate on.
        output_dir : str
            Path where snapshots will be written.
        overwrite : bool
            If `True`, write over existing output files if present.
            Otherwise, skip writing output files for a given transformation when already present.
            Assumes that for a given `run_id` the output files do not ever change;
            does *no* checking that files wouldn't be different if inputs for a given `run_id` have changed.


        Returns
        -------
        None
        """
        max_binding_free_energy = self.config.max_binding_free_energy

        # create output directory if not present
        run_id = transformation.transformation.run_id
        os.makedirs(os.path.join(output_dir, f"RUN{run_id}"), exist_ok=True)

        # TODO: Cache results and only update RUNs for which we have received new data to speed this up
        if (
            max_binding_free_energy is not None
            and transformation.binding_free_energy.point > max_binding_free_energy
        ):
            logging.info(
                "Skipping snapshot for RUN %d. Binding free energy estimate %g exceeds threshold %g",
                transformation.transformation.run_id,
                transformation.binding_free_energy.point,
                max_binding_free_energy,
            )
            return None

        gen_works = [
            (gen, work)
            for gen in transformation.complex_phase.gens
            for work in gen.works
        ]

        for ligand in ["old", "new"]:

            # check if output files all exist; if so, skip unless we are told not to
            if not overwrite:
                outfiles = self._transformation_to_file_mapping(
                    output_dir, run_id, ligand
                )
                if all(map(os.path.exists, outfiles)):
                    continue

            if ligand == "old":
                gen_work = min(gen_works, key=lambda gen_work: gen_work[1].reverse)
                frame = 3  # TODO: Magic numbers
            elif ligand == "new":
                gen_work = min(gen_works, key=lambda gen_work: gen_work[1].forward)
                frame = 1  # TODO: Magic numbers

            gen_analysis, workpair = gen_work

            # Extract representative snapshot
            import traceback

            try:
                sliced_snapshots, components = self.extract_snapshot(
                    project_dir=self.project_dir,
                    project_data_dir=self.project_data_dir,
                    ligand=ligand,
                    run=run_id,
                    clone=workpair.clone,
                    gen=gen_analysis.gen,
                    frame=frame,
                    fragment_id=transformation.transformation.xchem_fragment_id,
                )

                # Write protein PDB
                name = f"{ligand}_protein"
                sliced_snapshots["protein"].save(
                    os.path.join(output_dir, f"RUN{run_id}", f"{name}.pdb")
                )

                # Write old and new complex PDBs
                # TODO: Make sure that the atom names/elements are correctly interpolated for atoms shared between old/new
                # FIXME: This is producing incorrect new ligand complex PDBs, because some atoms are not changing identity
                name = f"{ligand}_complex"
                sliced_snapshots["complex"].save(
                    os.path.join(output_dir, f"RUN{run_id}", f"{name}.pdb")
                )

                # Write ligand SDFs
                # FIXME: This is producing incorrect new ligand SDFs
                from openeye import oechem

                name = f"{ligand}_ligand"
                with oechem.oemolostream(
                    os.path.join(output_dir, f"RUN{run_id}", f"{name}.sdf")
                ) as ofs:
                    oechem.OEWriteMolecule(ofs, components["ligand"])
            except Exception as e:
                print(
                    f"\nException occurred extracting snapshot from {self.project_dir} data {self.project_data_dir} run {run_id} clone {gen_work[1].clone} gen {gen_work[0].gen}"
                )
                print(e)
                traceback.print_exc()

    def generate_representative_snapshots(
        self,
        transformations: List[TransformationAnalysis],
        output_dir: str,
        num_procs: Optional[int],
        overwrite: bool = False,
    ) -> None:
        from rich.progress import track

        with multiprocessing.Pool(num_procs) as pool:
            result_iter = pool.imap_unordered(
                partial(
                    self.generate_representative_snapshot,
                    output_dir=output_dir,
                    overwrite=overwrite,
                ),
                transformations,
            )

            for _ in track(
                result_iter,
                total=len(transformations),
                description="Generating representative snapshots",
            ):
                pass
