import logging
from typing import Dict, List, Tuple

from ..schema import (
    CompoundAnalysis,
    CompoundMicrostate,
    CompoundSeriesAnalysis,
    Dict,
    MicrostateAnalysis,
)
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


def get_microstate_detail(
    compounds: List[CompoundAnalysis],
) -> Dict[CompoundMicrostate, Tuple[CompoundAnalysis, MicrostateAnalysis]]:
    return {
        CompoundMicrostate(
            compound_id=compound.metadata.compound_id,
            microstate_id=microstate.microstate.microstate_id,
        ): (
            compound,
            microstate,
        )
        for compound in compounds
        for microstate in compound.microstates
    }


def generate_report(series_analysis: CompoundSeriesAnalysis, results_path: str) -> None:
    """
    Postprocess results of calculations to extract summary for compound prioritization

    Parameters
    ----------
    analysis : Analysis
        Analysis results
    results_path : str
        Path to write results
    """

    import os

    structures_path = os.path.join(results_path, "structures")

    microstate_detail = get_microstate_detail(series_analysis.compounds)

    # Load all molecules, attaching properties
    # TODO: Generalize this to handle other than x -> 0 star map transformations
    from openeye import oechem
    from rich.progress import track

    oemols = list()  # target molecules
    refmols = list()  # reference molecules
    for analysis in track(
        series_analysis.transformations, description="Reading ligands"
    ):

        # Don't load anything not predicted to bind better
        if analysis.binding_free_energy.point >= 0.0:
            continue

        transformation = analysis.transformation
        run = f"RUN{transformation.run_id}"

        # Read target compound information
        protein_pdb_filename = os.path.join(structures_path, f"{run}-old_protein.pdb")
        ligand_sdf_filename = os.path.join(structures_path, f"{run}-old_ligand.sdf")

        # Read target compound
        oemol = oechem.OEMol()
        with oechem.oemolistream(ligand_sdf_filename) as ifs:
            oechem.OEReadMolecule(ifs, oemol)

        # Read reference compound
        refmol = oechem.OEMol()
        reference_ligand_sdf_filename = os.path.join(
            structures_path, f"{run}-new_ligand.sdf"
        )
        with oechem.oemolistream(reference_ligand_sdf_filename) as ifs:
            oechem.OEReadMolecule(ifs, refmol)
        refmols.append(refmol)

        # Set ligand title
        title = transformation.initial_microstate.microstate_id
        oemol.SetTitle(title)
        oechem.OESetSDData(oemol, "CID", title)

        # Set SMILES
        smiles = microstate_detail[transformation.initial_microstate].smiles
        oechem.OESetSDData(oemol, "SMILES", smiles)

        # Set RUN
        oechem.OESetSDData(oemol, "RUN", run)

        # Set free energy and uncertainty (in kcal/mol)
        # TODO: Improve this by writing appropriate digits of precision
        oechem.OESetSDData(
            oemol,
            "DDG (kcal/mol)",
            f"{KT_KCALMOL*analysis.binding_free_energy.stderr:.2f}",
        )
        oechem.OESetSDData(
            oemol,
            "dDDG (kcal/mol)",
            f"{KT_KCALMOL*analysis.binding_free_energy.stderr:.2f}",
        )

        # Store compound
        oemols.append(oemol)

    logging.info(f"{len(oemols)} molecules read")

    # Sort ligands in order of most favorable transformations
    import numpy as np

    sorted_indices = np.argsort(
        [float(oechem.OEGetSDData(oemol, "DDG (kcal/mol)")) for oemol in oemols]
    )

    # Filter based on threshold
    THRESHOLD = -0.5  # kcal/mol # TODO: expose as a parameter
    sorted_indices = [
        index
        for index in sorted_indices
        if (float(oechem.OEGetSDData(oemols[index], "DDG (kcal/mol)")) < -THRESHOLD)
    ]

    # Slice
    oemols = [oemols[index] for index in sorted_indices]
    refmols = [refmols[index] for index in sorted_indices]

    # Write sorted molecules
    for filename in ["ligands.sdf", "ligands.csv", "ligands.mol2"]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for oemol in track(oemols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, oemol)

    # Write PDF report
    write_pdf_report(
        oemols, os.path.join(results_path, "ligands.pdf"), series_analysis.metadata.name
    )

    # Write reference molecules
    for filename in ["reference.sdf", "reference.mol2"]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for refmol in track(refmols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, refmol)

    # Compile proteins
    import mdtraj as md
    import numpy as np

    proteins = list()
    for oemol in track(oemols, description="Consolidating protein snapshots"):
        RUN = oechem.OEGetSDData(oemol, "RUN")
        protein_pdb_filename = os.path.join(structures_path, f"{RUN}-old_protein.pdb")
        try:
            protein = md.load(protein_pdb_filename)
            proteins.append(protein)
        except IOError as e:
            logging.warning("Failed to load protein snapshot: %s", e)
            continue

    if not proteins:
        raise ValueError("No protein snapshots found")

    n_proteins = len(proteins)
    n_atoms = proteins[0].topology.n_atoms
    n_dim = 3
    xyz = np.zeros([n_proteins, n_atoms, n_dim], np.float32)
    for index, protein in enumerate(proteins):
        xyz[index, :, :] = protein.xyz[0, :, :]
    trajectory = md.Trajectory(xyz, proteins[0].topology)
    trajectory.save(os.path.join(results_path, "proteins.pdb"))
