"""
Prepare all SARS-CoV-2 Mpro structures for docking and simulation in monomer and dimer forms

This should be run from the covid-moonshot/scripts directory

"""
import os
from typing import Optional, List, Tuple, NamedTuple, Union
import re
from pathlib import Path
import argparse
import itertools
import tempfile
from argparse import ArgumentParser
import requests
from zipfile import ZipFile

from rich.progress import track
from openeye import oespruce
from openeye import oedocking
import numpy as np
from openeye import oechem

from prepare.constants import (BIOLOGICAL_SYMMETRY_HEADER, SEQRES_DIMER, SEQRES_MONOMER, FRAGALYSIS_URL,
                               MINIMUM_FRAGMENT_SIZE, CHAIN_PDB_INDEX)


class DockingSystem(NamedTuple):
    protein: oechem.OEGraphMol
    ligand: oechem.OEGraphMol
    receptor: oechem.OEGraphMol
    design_unit: oechem.OEDesignUnit


class PreparationConfig(NamedTuple):
    input: Path
    output: Path
    create_dimer: bool
    retain_water: Optional[bool] = False

    def __str__(self):
        msg = f"\n Input path: {str(self.input.absolute())}" \
              f"\n Output path: {str(self.output.absolute())}" \
              f"\n Create dimer: {str(self.create_dimer)}" \
              f"\n Retain water: {str(self.retain_water)}"
        return msg


class OutputPaths(NamedTuple):
    receptor_gzipped: Path
    receptor_thiolate_gzipped: Path
    design_unit_gzipped: Path
    design_unit_thiolate_gzipped: Path
    protein_pdb: Path
    protein_thiolate_pdb: Path
    ligand_pdb: Path
    ligand_sdf: Path
    ligand_mol2: Path


def download_url(url, save_path, chunk_size=128):
    """
    Download file from the specified URL to the specified file path, creating base dirs if needed.
    """
    # Create directory
    base_path, filename = os.path.split(save_path)
    os.makedirs(base_path, exist_ok=True)

    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        nchunks = int(int(r.headers['Content-Length'])/chunk_size)
        for chunk in track(r.iter_content(chunk_size=chunk_size), 'Downloading ZIP archive of Mpro structures...',
                           total=nchunks):
            fd.write(chunk)


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


def create_output_filenames(config: PreparationConfig) -> OutputPaths:
    prefix = config.output.joinpath(config.input.stem)
    stem = config.input.stem
    outputs = OutputPaths(
        # TODO: Can we simplify these paths by auto-generating them from a common prefix?
        receptor_gzipped=config.output.joinpath(f'{stem}-receptor.oeb.gz'),
        receptor_thiolate_gzipped=config.output.joinpath(f'{stem}-receptor-thiolate.oeb.gz'),
        design_unit_gzipped=config.output.joinpath(f'{stem}-designunit.oeb.gz'),
        design_unit_thiolate_gzipped=config.output.joinpath(f'{stem}-designunit-thiolate.oeb.gz'),
        protein_pdb=config.output.joinpath(f'{stem}-protein.pdb'),
        protein_thiolate_pdb=config.output.joinpath(f'{stem}-protein-thiolate.pdb'),
        ligand_pdb=config.output.joinpath(f'{stem}-ligand.pdb'),
        ligand_sdf=config.output.joinpath(f'{stem}-ligand.sdf'),
        ligand_mol2=config.output.joinpath(f'{stem}-ligand.mol2')
    )
    return outputs


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


def options_consistent(config: PreparationConfig) -> bool:
    flag = True
    if au_is_dimeric(config.input) and (not config.create_dimer):
        oechem.OEThrow.Verbose("Can't create monomer from dimer in AU")
        flag = False
    if (not au_is_dimeric(config.input)) and (crystal_series(config.input) in ['N', 'P']) and config.create_dimer:
        oechem.OEThrow.Verbose("Can't create dimer from monomer of N or P series")
        flag = False
    return flag


def create_logger(outputs: OutputPaths) -> oechem.oeofstream:
    fname = outputs.protein_pdb.parent.joinpath(f"{outputs.protein_pdb.stem}.log")
    errfs = oechem.oeofstream(str(fname)) # create a stream that writes internally to a stream
    oechem.OEThrow.SetOutputStream(errfs)
    oechem.OEThrow.Clear()
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Verbose) # capture verbose error output
    return errfs


# def align_complex()

def prepare_receptor(config: PreparationConfig) -> None:

    output_filenames = create_output_filenames(config)

    errfs = create_logger(output_filenames)

    if not options_consistent(config):
        oechem.OEThrow.Verbose(f"Options are inconsistent. Ignoring this configuration: \n{config}")
        errfs.close()
        return

    oechem.OEThrow.Verbose('cleaning pdb...')
    pdb_lines = clean_pdb(config)
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


def download_fragalysis_latest(structures_path: Path) -> None:
    zip_path = structures_path.joinpath('Mpro.zip')
    download_url(FRAGALYSIS_URL, zip_path)
    with ZipFile(zip_path, 'r') as zip_obj:
        zip_obj.extractall(structures_path)
    zip_path.unlink()


def get_structures(args: argparse.Namespace) -> List[Path]:
    structures_directory = args.structures_directory.absolute()
    if not structures_directory.exists() or not any(structures_directory.iterdir()):
        print(f"Downloading and extracting MPro files to {args.structures_directory.absolute()}")
        download_fragalysis_latest(args.structures_directory.absolute())


    source_pdb_files = list(structures_directory.glob(args.structures_filter))
    if len(source_pdb_files) == 0:
        raise RuntimeError(f'Glob path {structures_directory.joinpath(args.structures_filter)} '
                           f'has matched 0 files.')
    return source_pdb_files


def define_prep_configs(args: argparse.Namespace) -> List[PreparationConfig]:
    input_paths = get_structures(args)
    output_paths = [args.output_directory.absolute().joinpath(subdir) for subdir in ['monomer', 'dimer']]

    products = list(itertools.product(input_paths, output_paths))
    configs = [PreparationConfig(input=x, output=y, create_dimer=y.stem == 'dimer') for x, y in
               products]

    return configs


def create_output_directories(configs: List[PreparationConfig]) -> None:
    for config in configs:
        if config.output.exists():
            pass
        else:
            config.output.mkdir(parents=True, exist_ok=True)


def configure_parser(sub_subparser: ArgumentParser):
    p = sub_subparser.add_parser('receptors')
    p.add_argument('-i', '--structures-directory', type=Path,
                   help="Path to MPro directory, doesn't need to exist. Default: './MPro'",
                   default='./MPro')
    p.add_argument('-f', '--structures-filter', type=str,
                   help="Glob filter to find PDB files in structures_directory.",
                   default="aligned/Mpro-*_0?/Mpro-*_0?_bound.pdb")
    p.add_argument('-o', '--output-directory', type=Path, help='Path to directory in which to write prepared files',
                   default='./receptors')
    p.add_argument('-n', '--dry-run', help='Dry run: file locations will be printed to stdout.', action='store_true')
    p.set_defaults(func=main)


def main(args, parser) -> None:
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

    configs = define_prep_configs(args)
    create_output_directories(configs)
    for config in configs:
        if args.dry_run:
            print(config)
        else:
            prepare_receptor(config)
