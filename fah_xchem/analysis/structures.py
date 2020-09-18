"""
Methods for extracting snapshots and structures from core22 FAH trajectories.

Limitations:
* The reference structure (`natoms_reference`) must share the same atom ordering as the first `natoms_reference` atoms of the trajectory.
  For now, this means that the SpruceTK prepared structure (`Mpro-x10789_0_bound-protein-thiolate.pdb`) is used

Dependencies:
* mdtraj >= 1.9.4 (conda-forge)

"""

from functools import partial
import logging
import multiprocessing
import os
import tempfile
from typing import Dict, List, Optional

import joblib
import mdtraj as md

from ..schema import TransformationAnalysis


def load_trajectory(
    project_dir: str, project_data_dir: str, run: int, clone: int, gen: int
) -> md.Trajectory:
    """
    Load the trajectory from the specified PRCG.

    Parameters
    ----------
    project_dir : str
        Path to project directory (e.g. '/home/server/server2/projects/13422')
    project_data_dir : str
        Path to project data directory (e.g. '/home/server/server2/data/SVR314342810/PROJ13422')
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

    # Load trajectory
    pdbfile_path = os.path.join(project_dir, "RUNS", f"RUN{run}", "hybrid_complex.pdb")

    # TODO: Reuse path logic from fah_xchem.lib
    trajectory_path = os.path.join(
        project_data_dir,
        f"RUN{run}",
        f"CLONE{clone}",
        f"results{gen}",
        "positions.xtc",
    )
    try:
        pdbfile = md.load(pdbfile_path)
    except OSError as e:
        raise ValueError(f"Failed to load PDB file: {e}")

    try:
        trajectory = md.load(trajectory_path, top=pdbfile.top)
    except OSError as e:
        raise ValueError(f"Failed to load trajectory: {e}")

    return trajectory


def load_fragment(fragment_id: str) -> md.Trajectory:
    """
    Load the reference fragment structure

    Parameters
    ----------
    fragment_id : str
        Fragment ID (e.g. 'x10789')

    Returns
    -------
    fragment : mdtraj.Trajectory
        The fragment structure

    """

    # TODO: Put this in the covid-moonshot path, or generalize to an arbitrary file
    fragment = md.load(
        f"/home/server/server2/projects/available/covid-moonshot/receptors/monomer/Mpro-{fragment_id}_0_bound-protein-thiolate.pdb"
    )

    return fragment


def mdtraj_to_oemol(snapshot: md.Trajectory):
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
    project_dir: str,
    project_data_dir: str,
    run: int,
    clone: int,
    gen: int,
    frame: int,
    fragment_id: str,
    cache_dir: Optional[str],
):
    """
    Extract the specified snapshot, align it to the reference fragment, and write protein and ligands to separate PDB files

    Parameters
    ----------
    project_dir : str
       Path to project directory (e.g. '/home/server/server2/projects/13422')
    run : str or int
       Run (e.g. '0')
    clone : str or int
       Clone (e.g. '0')
    gen : str or int
       Gen (e.g. '0')
    frame : int
    fragment_id : str
      Fragment ID (e.g. 'x10789')
    cache_dir : str or None
       If specified, cache relevant parts of "htf.npz" file in a local directory of this name

    Returns
    -------
    sliced_snapshot : dict of str : mdtraj.Trajectory
      sliced_snapshot[name] is the Trajectory for name in ['protein', 'old_ligand', 'new_ligand', 'old_complex', 'new_complex']
    components : dict of str : oechem.OEMol
      components[name] is the OEMol for name in ['protein', 'old_ligand', 'new_ligand']

    """
    # Load the trajectory
    trajectory = load_trajectory(project_dir, project_data_dir, run, clone, gen)

    # Load the fragment
    fragment = load_fragment(fragment_id)

    # Align the trajectory to the fragment (in place)
    trajectory.image_molecules(inplace=True)
    trajectory.superpose(fragment, atom_indices=fragment.top.select("name CA"))

    # Extract the snapshot
    snapshot = trajectory[frame]

    # Slice out old or new state
    sliced_snapshot = slice_snapshot(snapshot, project_dir, run, cache_dir)

    # Convert to OEMol
    # NOTE: This uses heuristics, and should be replaced once we start storing actual chemical information
    components = dict()
    for name in ["protein", "old_ligand", "new_ligand"]:
        components[name] = mdtraj_to_oemol(sliced_snapshot[name])

    return sliced_snapshot, components


def get_stored_atom_indices(project_dir: str, run: int):
    """
    Load hybrid topology file and return relevant atom indices.
    """

    import numpy as np

    path = os.path.join(project_dir, "RUNS", f"RUN{run}")
    htf = np.load(os.path.join(path, "htf.npz"), allow_pickle=True)["arr_0"].tolist()

    # Determine mapping between hybrid topology and stored atoms in the positions.xtc
    # <xtcAtoms v="solute"/> eliminates waters
    nonwater_atom_indices = htf.hybrid_topology.select("not water")
    hybrid_to_stored_map = {
        nonwater_atom_indices[index]: index
        for index in range(len(nonwater_atom_indices))
    }

    # Get all atom indices from the hybrid system
    protein_atom_indices = htf.hybrid_topology.select("protein")
    hybrid_ligand_atom_indices = htf.hybrid_topology.select("resn MOL")

    # Identify atom index subsets for the old and new ligands from the hybrid system
    old_ligand_atom_indices = [
        index
        for index in hybrid_ligand_atom_indices
        if index in htf._old_to_hybrid_map.values()
    ]
    new_ligand_atom_indices = [
        index
        for index in hybrid_ligand_atom_indices
        if index in htf._new_to_hybrid_map.values()
    ]

    # Compute sliced atom indices using atom indices within positions.xtc
    return {
        "protein": [hybrid_to_stored_map[index] for index in protein_atom_indices],
        "old_ligand": [
            hybrid_to_stored_map[index] for index in old_ligand_atom_indices
        ],
        "new_ligand": [
            hybrid_to_stored_map[index] for index in new_ligand_atom_indices
        ],
        "old_complex": [
            hybrid_to_stored_map[index]
            for index in list(protein_atom_indices) + list(old_ligand_atom_indices)
        ],
        "new_complex": [
            hybrid_to_stored_map[index]
            for index in list(protein_atom_indices) + list(new_ligand_atom_indices)
        ],
    }


def slice_snapshot(
    snapshot: md.Trajectory,
    project_dir: str,
    run: int,
    cache_dir: Optional[str],
) -> Dict[str, md.Trajectory]:
    """
    Slice snapshot to specified state in-place

    .. TODO ::

       The htf.npz file is very slow to load.
       Replace this with a JSON file containing relevant ligand indices only

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
      sliced_snapshot[x] where x is one of ['protein', 'old_ligand', 'new_ligand', 'old_complex', 'new_complex']

    """

    get_stored_atom_indices_cached = (
        get_stored_atom_indices
        if cache_dir is None
        else joblib.Memory(cachedir=cache_dir, verbose=0).cache(get_stored_atom_indices)
    )

    stored_atom_indices = get_stored_atom_indices_cached(project_dir, run)

    sliced_snapshot = dict()
    for key, atom_indices in stored_atom_indices.items():
        sliced_snapshot[key] = md.Trajectory(
            snapshot.xyz[:, atom_indices, :], snapshot.topology.subset(atom_indices)
        )

    return sliced_snapshot


def generate_representative_snapshot(
    transformation: TransformationAnalysis,
    project_dir: str,
    project_data_dir: str,
    output_dir: str,
    max_binding_free_energy: Optional[float],
    cache_dir: Optional[str] = None,
) -> None:

    r"""
    Generate representative snapshots for old and new ligands.

    Illustration of frames:

    old ---[0]\             /[3]
               \           /
    new         \[1]---[2]/

    Parameters
    ----------
    project_dir : str
        Path to project directory (e.g. '/home/server/server2/projects/13422')
    project_data_dir : str
        Path to project data directory (e.g. '/home/server/server2/data/SVR314342810/PROJ13422')
    run : int
        Run (e.g. '0')
    works : list of WorkPair
        Work values extracted from simulation results
    fragment_id : str
        Fragment ID (e.g. 'x10789')
    snapshot_output_path: str
        Path where snapshots will be written
    cache_dir : str or None, optional
        If specified, cache relevant parts of "htf.npz" file in a local directory of this name

    Returns
    -------
    None
    """

    if (
        max_binding_free_energy is not None
        and transformation.binding_free_energy.point > max_binding_free_energy
    ):
        logging.warning(
            "Skipping snapshot for RUN %d. Binding free energy estimate %g exceeds threshold %g",
            transformation.transformation.run_id,
            transformation.binding_free_energy.point,
            max_binding_free_energy,
        )
        return None

    gen_works = [
        (gen, work) for gen in transformation.complex_phase.gens for work in gen.works
    ]

    for ligand in ["old", "new"]:
        if ligand == "old":
            gen_work = min(gen_works, key=lambda gen_work: gen_work[1].reverse)
            frame = 3  # TODO: Magic numbers
        else:
            gen_work = min(gen_works, key=lambda gen_work: gen_work[1].forward)
            frame = 1  # TODO: Magic numbers

        run_id = transformation.transformation.run_id

        # Extract representative snapshot
        sliced_snapshots, components = extract_snapshot(
            project_dir=project_dir,
            project_data_dir=project_data_dir,
            run=run_id,
            clone=gen_work[1].clone,
            gen=gen_work[0].gen,
            frame=frame,
            fragment_id=transformation.transformation.xchem_fragment_id,
            cache_dir=cache_dir,
        )

        # Write protein PDB
        name = f"{ligand}_protein"
        sliced_snapshots["protein"].save(
            os.path.join(output_dir, f"RUN{run_id}-{name}.pdb")
        )

        # Write old and new complex PDBs
        name = f"{ligand}_complex"
        sliced_snapshots[name].save(os.path.join(output_dir, f"RUN{run_id}-{name}.pdb"))

        # Write ligand SDFs
        from openeye import oechem

        name = f"{ligand}_ligand"
        with oechem.oemolostream(
            os.path.join(output_dir, f"RUN{run_id}-{name}.sdf")
        ) as ofs:
            oechem.OEWriteMolecule(ofs, components[name])


def generate_representative_snapshots(
    transformations: List[TransformationAnalysis],
    project_dir: str,
    project_data_dir: str,
    output_dir: str,
    max_binding_free_energy: Optional[float],
    cache_dir: Optional[str],
    num_procs: Optional[int],
) -> None:
    from rich.progress import track

    with multiprocessing.Pool(num_procs) as pool:
        result_iter = pool.imap_unordered(
            partial(
                generate_representative_snapshot,
                project_dir=project_dir,
                project_data_dir=project_data_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
                max_binding_free_energy=max_binding_free_energy,
            ),
            transformations,
        )
        track(
            result_iter,
            total=len(transformations),
            description="Generating representative snapshots",
        )
