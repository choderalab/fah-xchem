import logging
from typing import List, Optional

from ..schema import CompoundMicrostate, CompoundSeriesAnalysis, TransformationAnalysis
from .constants import KT_KCALMOL

def write_pdf_report(mollist, pdf_filename, iname):
    """
    Write molecules with SD Data to PDF

    Parameters
    ----------
    mollist : list of openeye.oechem.OEMol
        The list of molecules with SD data tags
    pdf_filename : str
        The PDF filename to write
    iname : str
        Dataset name

    """
    from openeye import oedepict

    # collect data tags
    tags = CollectDataTags(mollist)

    # initialize multi-page report
    rows, cols = 4, 2
    ropts = oedepict.OEReportOptions(rows, cols)
    ropts.SetHeaderHeight(25)
    ropts.SetFooterHeight(25)
    ropts.SetCellGap(2)
    ropts.SetPageMargins(10)
    report = oedepict.OEReport(ropts)

    # setup depiction options
    cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
    opts = oedepict.OE2DMolDisplayOptions(
        cellwidth, cellheight, oedepict.OEScale_AutoScale
    )

    # generate report
    DepictMoleculesWithData(report, mollist, iname, tags, opts)
    oedepict.OEWriteReport(pdf_filename, report)


def CollectDataTags(mollist):
    from openeye import oechem

    tags = []
    for mol in mollist:
        for dp in oechem.OEGetSDDataIter(mol):
            if not dp.GetTag() in tags:
                tags.append(dp.GetTag())

    return tags


def DepictMoleculesWithData(report, mollist, iname, tags, opts):
    from openeye import oechem
    from openeye import oedepict

    for mol in mollist:
        # render molecule
        cell = report.NewCell()
        oedepict.OEPrepareDepiction(mol)
        disp = oedepict.OE2DMolDisplay(mol, opts)
        oedepict.OERenderMolecule(cell, disp)
        oedepict.OEDrawCurvedBorder(cell, oedepict.OELightGreyPen, 10.0)

        # render corresponding data
        cell = report.NewCell()
        RenderData(cell, mol, tags)

    # add input filnename to headers
    headerfont = oedepict.OEFont(
        oedepict.OEFontFamily_Default,
        oedepict.OEFontStyle_Default,
        12,
        oedepict.OEAlignment_Center,
        oechem.OEBlack,
    )
    headerpos = oedepict.OE2DPoint(
        report.GetHeaderWidth() / 2.0, report.GetHeaderHeight() / 2.0
    )

    for header in report.GetHeaders():
        header.DrawText(headerpos, iname, headerfont)

    # add page number to footers
    footerfont = oedepict.OEFont(
        oedepict.OEFontFamily_Default,
        oedepict.OEFontStyle_Default,
        12,
        oedepict.OEAlignment_Center,
        oechem.OEBlack,
    )
    footerpos = oedepict.OE2DPoint(
        report.GetFooterWidth() / 2.0, report.GetFooterHeight() / 2.0
    )

    for pageidx, footer in enumerate(report.GetFooters()):
        footer.DrawText(footerpos, "- %d -" % (pageidx + 1), footerfont)


def RenderData(image, mol, tags):
    from openeye import oechem
    from openeye import oedepict

    data = []
    for tag in tags:
        value = "N/A"
        if oechem.OEHasSDData(mol, tag):
            value = oechem.OEGetSDData(mol, tag)
        data.append((tag, value))

    nrdata = len(data)

    tableopts = oedepict.OEImageTableOptions(
        nrdata, 2, oedepict.OEImageTableStyle_LightBlue
    )
    tableopts.SetColumnWidths([10, 20])
    tableopts.SetMargins(2.0)
    tableopts.SetHeader(False)
    tableopts.SetStubColumn(True)
    table = oedepict.OEImageTable(image, tableopts)

    for row, (tag, value) in enumerate(data):
        cell = table.GetCell(row + 1, 1)
        table.DrawText(cell, tag + ":")
        cell = table.GetBodyCell(row + 1, 1)
        table.DrawText(cell, value)

def gens_are_consistent(
        transformation: TransformationAnalysis,
        ngens: Optional[int] = 2,
        nsigma: Optional[float] = 3,
) -> bool:
    """
    Return True if GENs are consistent.
    
    The last `ngens` generations will be checked for consistency with the overall estimate,
    and those with estimates that deviate by more than `nsigma` standard errors will be dropped.
sprint-5-minimal-test.json
    Parameters
    ----------
    transformation : TransformationAnalysis
        The TransformationAnalysis object to check for consistency
    ngens : int, optional, default=2
        The last `ngens` generations will be checked for consistency with the overall estimate
    nsigma : int, optional, default=3
        Number of standard errors of overall estimate to use for consistency check
    """
    # Collect free energy estimates for each GEN
    ngens = min( len(transformation.complex_phase.gens), len(transformation.solvent_phase.gens) )
    gen_estimates = list()
    for gen in range(ngens):
        complex_delta_f = transformation.complex_phase.gens[gen].free_energy.delta_f
        solvent_delta_f = transformation.solvent_phase.gens[gen].free_energy.delta_f
        if (complex_delta_f is None) or (solvent_delta_f is None):
            continue
        binding_delta_f = complex_delta_f - solvent_delta_f
        gen_estimates.append(binding_delta_f)
        
    if len(gen_estimates) < ngens:
        # We don't have enough GENs
        return False

    # Flag inconsistent if any GEN estimate is more than nsigma stderrs away from overall estimate
    for gen_delta_f in gen_estimates[-ngens:]:
        overall_delta_f = transformation.binding_free_energy
        delta_f = overall_delta_f - gen_delta_f
        #print(gen_delta_f, overall_delta_f, delta_f)
        #if abs(delta_f.point) > nsigma*delta_f.stderr:
        if abs(delta_f.point) > nsigma*gen_delta_f.stderr:
            return False
        
    return True

def generate_report(
        series: CompoundSeriesAnalysis,
        results_path: str,
        max_binding_free_energy: float=0.0,
        consolidate_protein_snapshots: Optional[bool] = True,
        filter_gen_consistency: Optional[bool] = True,
) -> None:
    """
    Postprocess results of calculations to extract summary for compound prioritization

    Parameters
    ----------
    series : CompoundSeriesAnalysis
        Analysis results
    results_path : str
        Path to write results
    max_binding_free_energy : str, optional, default=0
        Don't report compounds with free energies greater than this (in kT)
    consolidate_protein_snapshots : bool, optional, default=True
        If True, consolidate all protein snapshots into a single PDB file
    """

    import os

    if filter_gen_consistency:
        logging.info(f"Filtering transformations for GEN-to-GEN consitency...")

    # Load all molecules, attaching properties
    # TODO: Generalize this to handle other than x -> 0 star map transformations
    from openeye import oechem
    from rich.progress import track

    microstate_detail = {
        CompoundMicrostate(
            compound_id=compound.metadata.compound_id,
            microstate_id=microstate.microstate.microstate_id,
        ): microstate.microstate
        for compound in series.compounds
        for microstate in compound.microstates
    }

    # TODO: Take this cutoff from global configuration
    oemols = list()  # target molecules
    refmols = list()  # reference molecules
    # TODO : Iterate over compounds instead of transformations
    # Store optimal microstates for each compound, and representative snapshot paths for each microstate and compound in analysis
    for transformation in track(series.transformations, description="Reading ligands"):

        # Don't bother reading ligands with transformation free energies above max
        # since snapshots aren't generated for these
        if transformation.binding_free_energy.point >= max_binding_free_energy:
            continue

        # Filter by consistency of GENs if requested
        if filter_gen_consistency:
            if not gens_are_consistent(transformation):
                continue        
            # Label current iteration as reliable (JSON boolean)
            transformation.transformation.reliable_transform = "true"

        run = f"RUN{transformation.transformation.run_id}"
        path = os.path.join(results_path, "transformations", run)

        # Read target compound information
        protein_pdb_filename = os.path.join(path, "new_protein.pdb")
        ligand_sdf_filename = os.path.join(path, "new_ligand.sdf")

        # Read target compound
        oemol = oechem.OEMol()
        with oechem.oemolistream(ligand_sdf_filename) as ifs:
            oechem.OEReadMolecule(ifs, oemol)

        # Read reference compound
        refmol = oechem.OEMol()
        reference_ligand_sdf_filename = os.path.join(path, "old_ligand.sdf")
        with oechem.oemolistream(reference_ligand_sdf_filename) as ifs:
            oechem.OEReadMolecule(ifs, refmol)
        refmols.append(refmol)

        # Set ligand title
        title = transformation.transformation.final_microstate.microstate_id
        oemol.SetTitle(title)
        oechem.OESetSDData(oemol, "CID", title)

        # Set SMILES
        smiles = microstate_detail[
            transformation.transformation.final_microstate
        ].smiles
        oechem.OESetSDData(oemol, "SMILES", smiles)

        # Set RUN
        oechem.OESetSDData(oemol, "RUN", run)

        # Set free energy and uncertainty (in kcal/mol)
        # TODO: Improve this by writing appropriate digits of precision
        oechem.OESetSDData(
            oemol,
            "DDG (kcal/mol)",
            f"{KT_KCALMOL*transformation.binding_free_energy.point:.2f}",
        )
        oechem.OESetSDData(
            oemol,
            "dDDG (kcal/mol)",
            f"{KT_KCALMOL*transformation.binding_free_energy.stderr:.2f}",
        )

        # Store compound
        oemols.append(oemol)

    logging.info(f"{len(oemols)} molecules read")

    # Sort ligands in order of most favorable transformations
    import numpy as np

    logging.info(f"Sorting molecules to prioritize most favorable transformations")
    sorted_indices = np.argsort(
        [float(oechem.OEGetSDData(oemol, "DDG (kcal/mol)")) for oemol in oemols]
    )

    # Filter based on threshold
    sorted_indices = [
        index
        for index in sorted_indices
        if (float(oechem.OEGetSDData(oemols[index], "DDG (kcal/mol)")) < max_binding_free_energy)
    ]

    # Slice
    oemols = [oemols[index] for index in sorted_indices]
    refmols = [refmols[index] for index in sorted_indices]

    logging.info(f"{len(oemols)} molecules remain after filtering based on {max_binding_free_energy} threshold")

    # Write sorted molecules
    for filename in ["transformations-final-ligands.sdf", "transformations-final-ligands.csv", "transformations-final-ligands.mol2"]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for oemol in track(oemols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, oemol)

    # Write PDF report
    write_pdf_report(
        oemols, os.path.join(results_path, "transformations-final-ligands.pdf"), series.metadata.name
    )

    # Write reference molecules
    for filename in ["transformations-initial-ligands.sdf", "transformations-initial-ligands.mol2"]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for refmol in track(refmols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, refmol)

    if consolidate_protein_snapshots:
        consolidate_protein_snapshots_into_pdb(oemols, results_path)
        
from openeye import oechem
def consolidate_protein_snapshots_into_pdb(
        oemols: List[oechem.OEMol],
        results_path: str,
        pdb_filename: Optional[str] = 'transformations-final-proteins.pdb'
):
    """
    Consolidate protein snapshots into a single file

    Parameters
    ----------
    oemols : list of OEMol
        List of annotated OEMols
    results_path : str
        Analysis results path
    pdb_filename : str, optional, default='transformations-final-proteins.pdb'
        Filename (without path) to write compiled PDB file to
    """
    import mdtraj as md
    import numpy as np
    import os

    # TODO: Replace this with something that writes models as we read them
    # since this is highly memory inefficient and slow
        
    proteins = list()
    from rich.progress import track
    for oemol in track(oemols, description="Reading protein snapshots"):
        RUN = oechem.OEGetSDData(oemol, "RUN")
        protein_pdb_filename = os.path.join(
            results_path, "transformations", RUN, "old_protein.pdb"
        )
        try:
            protein = md.load(protein_pdb_filename)
            proteins.append(protein)
        except IOError as e:
            logging.warning("Failed to load protein snapshot: %s", e)
            continue

    if not proteins:
        return # DEBUG
        raise ValueError("No protein snapshots found")

    logging.info(f"Writing consolidated snapshots to {pdb_filename}")
    n_proteins = len(proteins)
    n_atoms = proteins[0].topology.n_atoms
    n_dim = 3
    xyz = np.zeros([n_proteins, n_atoms, n_dim], np.float32)
    for index, protein in enumerate(proteins):
        xyz[index, :, :] = protein.xyz[0, :, :]
    trajectory = md.Trajectory(xyz, proteins[0].topology)
    trajectory.save(os.path.join(results_path, pdb_filename))
