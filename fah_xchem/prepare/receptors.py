"""
Prepare all SARS-CoV-2 Mpro structures for docking and simulation in monomer and dimer forms

This should be run from the covid-moonshot/scripts directory

"""
import os
import pathlib
from typing import Optional, List, Tuple, NamedTuple, Union
import re
from pathlib import Path
import argparse
import itertools
import tempfile
from argparse import ArgumentParser
import requests
from zipfile import ZipFile

from openeye import oespruce
from openeye import oedocking
import numpy as np
from openeye import oechem
from pydantic import BaseModel, Field

from .prepare.constants import (BIOLOGICAL_SYMMETRY_HEADER, SEQRES_DIMER, SEQRES_MONOMER, FRAGALYSIS_URL,
                               MINIMUM_FRAGMENT_SIZE, CHAIN_PDB_INDEX)
from ..schema import OutputPaths, DockingSystem


class ReceptorArtifactory(BaseModel):
    """Receptor F@H-input creator.

    """
    input: pathlib.Path = Field(..., description="")
    output: pathlib.Path = Field(..., description="")
    create_dimer: bool = Field(..., description="")
    retain_water: Optional[bool] = Field(False, description="")

    def prepare_receptor(self) -> None:
    
        output_filenames = self._create_output_filenames()
    
        errfs = self._create_logger(output_filenames)
    
        if not self._options_consistent():
            oechem.OEThrow.Verbose(f"Options are inconsistent. Ignoring this configuration: \n{self.config}")
            errfs.close()
            return
    
        oechem.OEThrow.Verbose('cleaning pdb...')
        pdb_lines = clean_pdb(self.config)
        oechem.OEThrow.Verbose('creating molecular graph...')
        complex = lines_to_mol_graph(pdb_lines)
    
        # Some prophylatic measures by JDC - may be redundant in new OE versions
        oechem.OEThrow.Verbose('stripping hydrogens...')
        complex = strip_hydrogens(complex)
        oechem.OEThrow.Verbose('rebuilding c terminal...')
        complex = rebuild_c_terminal(complex)
    
        # Log warnings
    
    
        # Setup options
        oechem.OEThrow.Verbose('setting options...')
        options = set_options()
        oechem.OEThrow.Verbose('creating metadata...')
        metadata = oespruce.OEStructureMetadata()
    
        oechem.OEThrow.Verbose('making base design unit...')
        design_unit = make_design_units(complex, metadata, options)[0]
        oechem.OEThrow.Verbose('making base docking system...')
        docking_sytem = make_docking_system(design_unit)
    
        # Neutral dyad
        oechem.OEThrow.Verbose('making neutral dyad...')
        oechem.OEThrow.Verbose('\t updating design unit...')
        design_unit = create_dyad('His41(0) Cys145(0)', docking_sytem, design_unit, options)
        oechem.OEThrow.Verbose('\t updating docking system...')
        docking_sytem = make_docking_system(design_unit)
        oechem.OEThrow.Verbose('\t writing docking system...')
        write_docking_system(docking_sytem, output_filenames, is_thiolate=False)
    
        # Charge separated dyad
        oechem.OEThrow.Verbose('making catalytic dyad...')
        oechem.OEThrow.Verbose('\t updating design unit...')
        design_unit = create_dyad('His41(+) Cys145(-1)', docking_sytem, design_unit, options)
        oechem.OEThrow.Verbose('\t updating docking system...')
        docking_system = make_docking_system(design_unit)
        oechem.OEThrow.Verbose('\t writing docking system...')
        write_docking_system(docking_system, output_filenames, is_thiolate=True)
        errfs.close()


    def _create_output_filenames(self) -> OutputPaths:
        output = self.output
        prefix = output.joinpath(self.input.stem)
        stem = self.input.stem

        outputs = OutputPaths(
            # TODO: Can we simplify these paths by auto-generating them from a common prefix?
            receptor_gzipped=output.joinpath(f'{stem}-receptor.oeb.gz'),
            receptor_thiolate_gzipped=output.joinpath(f'{stem}-receptor-thiolate.oeb.gz'),
            design_unit_gzipped=output.joinpath(f'{stem}-designunit.oeb.gz'),
            design_unit_thiolate_gzipped=output.joinpath(f'{stem}-designunit-thiolate.oeb.gz'),
            protein_pdb=output.joinpath(f'{stem}-protein.pdb'),
            protein_thiolate_pdb=output.joinpath(f'{stem}-protein-thiolate.pdb'),
            ligand_pdb=output.joinpath(f'{stem}-ligand.pdb'),
            ligand_sdf=output.joinpath(f'{stem}-ligand.sdf'),
            ligand_mol2=output.joinpath(f'{stem}-ligand.mol2')
        )
        return outputs


    def _create_logger(self, outputs: OutputPaths) -> oechem.oeofstream:
        fname = outputs.protein_pdb.parent.joinpath(f"{outputs.protein_pdb.stem}.log")
        errfs = oechem.oeofstream(str(fname)) # create a stream that writes internally to a stream
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.Clear()
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Verbose) # capture verbose error output
        return errfs

    def _options_consistent(self) -> bool:
        flag = True
        if au_is_dimeric(self.input) and (not self.create_dimer):
            oechem.OEThrow.Verbose("Can't create monomer from dimer in AU")
            flag = False
        if (not au_is_dimeric(self.input)) and (crystal_series(self.input) in ['N', 'P']) and self.create_dimer:
            oechem.OEThrow.Verbose("Can't create dimer from monomer of N or P series")
            flag = False
        return flag


def read_pdb_file(pdb_file):

    ifs = oechem.oemolistream()
    ifs.SetFlavor(oechem.OEFormat_PDB, oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA | oechem.OEIFlavor_PDB_ALTLOC)  # noqa

    if not ifs.open(pdb_file):
        oechem.OEThrow.Fatal("Unable to open %s for reading." % pdb_file)

    mol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s." % pdb_file)
    ifs.close()

    return mol


def crystal_series(pdb_path: Path) -> str:
    if re.search(r'-x\d+', str(pdb_path.stem)) is not None:
        return 'x'
    if re.search(r'-N\d+', str(pdb_path.stem)) is not None:
        return 'N'
    if re.search(r'-P\d+', str(pdb_path.stem)) is not None:
        return 'P'
    else:
        raise ValueError('Type of crystal form not found.')


def remove_from_lines(lines: List[str], string: str) -> List[str]:
    return [line for line in lines if string not in line]


def is_in_lines(lines: List[str], string: str) -> bool:
    return np.any([string in line for line in lines])


def add_prefix(lines: List[str], prefix: str) -> List[str]:
    return [line + '\n' for line in prefix.split('\n')] + lines


def lines_to_mol_graph(lines: List[str]) -> oechem.OEGraphMol:

    with tempfile.NamedTemporaryFile(delete=False, mode='wt', suffix='.pdb') as pdbfile:
        pdbfile.write(''.join(lines))
        pdbfile.close()
        complex = read_pdb_file(pdbfile.name)

    return complex


def get_chain_labels_pdb(pdb_path: Path) -> List[str]:
    pdb_lines = pdb_path.open('rt').readlines()
    return list(set([x[CHAIN_PDB_INDEX].lower() for x in pdb_lines if x.startswith('ATOM')]))


def au_is_dimeric(pdb_path: Path) -> bool:
    labels = get_chain_labels_pdb(pdb_path)
    return ('a' in labels) and ('b' in labels)


def clean_pdb(config: PreparationConfig) -> List[str]:
    with config.input.open() as f:
        pdbfile_lines = f.readlines()

    if (crystal_series(config.input) == 'x') and (not is_in_lines(pdbfile_lines, 'REMARK 350')):
        pdbfile_lines = add_prefix(pdbfile_lines, BIOLOGICAL_SYMMETRY_HEADER)

    if not config.create_dimer:
        pdbfile_lines = remove_from_lines(pdbfile_lines, 'REMARK 350')

    if not config.retain_water:
        pdbfile_lines = remove_from_lines(pdbfile_lines, 'HOH')

    pdbfile_lines = remove_from_lines(pdbfile_lines, 'LINK')
    pdbfile_lines = remove_from_lines(pdbfile_lines, 'UNK')

    if not is_in_lines(pdbfile_lines, 'SEQRES'):
        if au_is_dimeric(config.input):
            pdbfile_lines = add_prefix(pdbfile_lines, SEQRES_DIMER)
        else:
            pdbfile_lines = add_prefix(pdbfile_lines, SEQRES_MONOMER)


    return pdbfile_lines


def strip_hydrogens(complex: oechem.OEGraphMol) -> oechem.OEGraphMol:
    for atom in complex.GetAtoms():
        if atom.GetAtomicNum() > 1:
            oechem.OESuppressHydrogens(atom)
    return complex


def already_prepared(outputs: OutputPaths) -> bool:
    return np.all([x.exists() for x in outputs])


def rebuild_c_terminal(complex: oechem.OEGraphMol) -> oechem.OEGraphMol:
    # Delete and rebuild C-terminal residue because Spruce causes issues with this
    # See: 6m2n 6lze
    pred = oechem.OEIsCTerminalAtom()
    for atom in complex.GetAtoms():
        if pred(atom):
            for nbor in atom.GetAtoms():
                if oechem.OEGetPDBAtomIndex(nbor) == oechem.OEPDBAtomName_O:
                    complex.DeleteAtom(nbor)
    return complex


def set_options() -> oespruce.OEMakeDesignUnitOptions:
    # Both N- and C-termini should be zwitterionic
    # Mpro cleaves its own N- and C-termini
    # See https://www.pnas.org/content/113/46/12997
    # Don't allow truncation of termini, since force fields don't have parameters for this
    # Build loops and sidechains

    opts = oespruce.OEMakeDesignUnitOptions()

    opts.GetSplitOptions().SetMinLigAtoms(MINIMUM_FRAGMENT_SIZE) # minimum fragment size (in heavy atoms)
    opts.GetPrepOptions().SetStrictProtonationMode(True)

    opts.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    opts.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
    opts.GetPrepOptions().GetBuildOptions().SetBuildLoops(True)
    opts.GetPrepOptions().GetBuildOptions().SetBuildSidechains(True)

    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(False)

    # Premptive measure by JDC - not sure whether this is actually needed.
    opts = prevent_flip(opts, match_strings=["GLN:189:.*:.*:.*"])

    return opts


def prevent_flip(options: oespruce.OEMakeDesignUnitOptions, match_strings: List[str]) -> oespruce.OEMakeDesignUnitOptions:
    pred = oechem.OEAtomMatchResidue(match_strings)
    protonate_opts = options.GetPrepOptions().GetProtonateOptions()
    place_hydrogens_opts = protonate_opts.GetPlaceHydrogensOptions()
    place_hydrogens_opts.SetNoFlipPredicate(pred)
    return options


def make_design_units(complex: oechem.OEGraphMol, metadata: oespruce.OEStructureMetadata,
                      options: oespruce.OEMakeDesignUnitOptions) -> List[oechem.OEDesignUnit]:
    dus = list(oespruce.OEMakeDesignUnits(complex, metadata, options))
    if len(dus) == 0:
        raise RuntimeError('No design units found.')
    return dus


def make_docking_system(design_unit: oechem.OEDesignUnit) -> DockingSystem:
    # Make a deep copy of the design unit so it isn't accidentally modified
    import copy
    design_unit = copy.deepcopy(design_unit)

    protein = oechem.OEGraphMol()
    design_unit.GetProtein(protein)

    ligand = oechem.OEGraphMol()
    design_unit.GetLigand(ligand)

    receptor = oechem.OEGraphMol()
    oedocking.OEMakeReceptor(receptor, protein, ligand)

    system = DockingSystem(protein=protein,
                           ligand=ligand,
                           receptor=receptor,
                           design_unit=design_unit)
    return system


def write_receptor(receptor: oechem.OEGraphMol, paths: List[Path]) -> None:
    for path in paths:
        # if not path.exists():
        oedocking.OEWriteReceptorFile(oechem.OEGraphMol(receptor), str(path))


def write_design_unit(design_unit: oechem.OEDesignUnit, paths: List[Path]) -> None:
    for path in paths:
        # if not path.exists():
        ofs = oechem.oeofstream(str(path))
        oechem.OEWriteDesignUnit(ofs, design_unit)
        ofs.close()


def write_protein(protein: oechem.OEGraphMol, paths: List[Path]) -> None:
    for path in paths:
        # if not path.exists():
        with oechem.oemolostream(str(path)) as ofs:
            oechem.OEWriteMolecule(ofs, oechem.OEGraphMol(protein))

        with path.open(mode='rt') as f:
            pdbfile_lines = f.readlines()

        pdbfile_lines = remove_from_lines(pdbfile_lines, 'UNK')

        with path.open(mode='wt') as outfile:
            outfile.write(''.join(pdbfile_lines))


def write_molecular_graph(molecule: oechem.OEGraphMol, paths: List[Path]) -> None:
    for path in paths:
        # if not path.exists():
        with oechem.oemolostream(str(path)) as ofs:
            oechem.OEWriteMolecule(ofs, oechem.OEGraphMol(molecule))


def write_docking_system(docking_system: DockingSystem, filenames: OutputPaths,
                         is_thiolate: Optional[bool] = False) -> None:
    if is_thiolate:
        design_unit_path = filenames.design_unit_thiolate_gzipped
        receptor_path = filenames.receptor_thiolate_gzipped
        protein_path = filenames.protein_thiolate_pdb
    else:
        design_unit_path = filenames.design_unit_gzipped
        receptor_path = filenames.receptor_gzipped
        protein_path = filenames.protein_pdb

    write_design_unit(design_unit=docking_system.design_unit, paths=[design_unit_path])
    write_receptor(receptor=docking_system.receptor, paths=[receptor_path])
    write_protein(protein=docking_system.protein, paths=[protein_path])
    paths = [filenames.ligand_mol2, filenames.ligand_pdb, filenames.ligand_sdf]
    write_molecular_graph(molecule=docking_system.ligand, paths=paths)


def get_atoms(molecule: oechem.OEGraphMol, match_string: str, atom_name: str) -> List[oechem.OEAtomBase]:
    atoms = []
    pred = oechem.OEAtomMatchResidue(match_string)
    for atom in molecule.GetAtoms(pred):
        if atom.GetName().strip() == atom_name:
            atoms.append(atom)
    return atoms


def bypass_atoms(match_strings: List[str], options: oechem.OEPlaceHydrogensOptions) -> oechem.OEPlaceHydrogensOptions:
    pred = oechem.OEAtomMatchResidue(match_strings)
    options.SetBypassPredicate(pred)
    return options


def create_dyad(state: str, docking_system: DockingSystem, design_unit: oechem.OEDesignUnit,
                    options: oespruce.OEMakeDesignUnitOptions) -> oechem.OEDesignUnit:

    protonate_opts = options.GetPrepOptions().GetProtonateOptions()
    place_h_opts = protonate_opts.GetPlaceHydrogensOptions()
    protein = docking_system.protein

    if state == 'His41(+) Cys145(-1)':
        atoms = get_atoms(protein, "CYS:145:.*:.*:.*", "SG")
        for atom in atoms:
            if atom.GetExplicitHCount() == 1:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(0)
                atom.SetFormalCharge(-1)

        atoms = get_atoms(protein, "HIS:41:.*:.*:.*", "ND1")
        for atom in atoms:
            if atom.GetExplicitHCount() == 0:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(1)
                atom.SetFormalCharge(+1)

        atoms = get_atoms(protein, "HIS:41:.*:.*:.*", "NE2")
        for atom in atoms:
            if atom.GetExplicitHCount() == 0:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(1)
                atom.SetFormalCharge(+1)

    elif state == 'His41(0) Cys145(0)':
        atoms = get_atoms(protein, "CYS:145:.*:.*:.*", "SG")
        for atom in atoms:
            if atom.GetExplicitHCount() == 0:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(1)
                atom.SetFormalCharge(0)

        atoms = get_atoms(protein, "HIS:41:.*:.*:.*", "ND1")
        for atom in atoms:
            if atom.GetFormalCharge() == 1:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(0)
                atom.SetFormalCharge(0)

        atoms = get_atoms(protein, "HIS:41:.*:.*:.*", "NE2")
        for atom in atoms:
            if atom.GetFormalCharge() == 1:
                oechem.OESuppressHydrogens(atom)  # strip hydrogens from residue
                atom.SetImplicitHCount(0)
                atom.SetFormalCharge(0)

    else:
        ValueError("dyad_state must be one of ['His41(0) Cys145(0)', 'His41(+) Cys145(-)']")

    place_h_opts = bypass_atoms(["HIS:41:.*:.*:.*", "CYS:145:.*:.*:.*"], place_h_opts)
    oechem.OEAddExplicitHydrogens(protein)
    oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)
    oespruce.OEProtonateDesignUnit(design_unit, protonate_opts)
    return design_unit
