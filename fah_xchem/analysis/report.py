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


def upload_fragalyis(
    ligand_filename: str,
    proteins_filename: str,
    ref_url: str,
    target_name: str,
    ref_mols: str,
    ref_pdb: str,
    method: str,
):

    """
    Generate input for fragalysis from ligand_filename and proteins_filename​

    Fragalysis spec:https://discuss.postera.ai/t/providing-computed-poses-for-others-to-look-at/1155/8?u=johnchodera​

    Parameters
    ----------
    ligand_filename : str
        The name of the ligand file to upload. An SDF file.
    pdf_filename : str
        The name of the protein file to upload. An PDB file.
    ref_url : str
        URL to postera.ai/covid forum post
    target_name : str
        The fragalysis dataset name e.g. 'foldingathome-sprint-4'
    ref_mols : str
        A comma separated list of the fragments that inspired the design of the new molecule (codes as they appear in fragalysis - e.g. 'x0104_0,x0692_0')
    ref_pdb : str
        The name of the fragment (and corresponding Mpro fragment structure) with the best scoring hybrid docking pose
    method : str
        The name for the method used to generate the compound poses (e.g. 'Sprint 4')

    """

    from openeye import oechem
    from rich.progress import track

    fragalysis_sdf_filename = f"{target_name}.sdf"
    submitter_name = "Folding@home"

    # Read ligand poses
    molecules = list()

    with oechem.oemolistream(ligands_filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            molecules.append(oemol.CreateCopy())
    print(f"{len(molecules)} ligands read")

    descriptions = {
        "DDG (kcal/mol)": "Relative computed free energy difference",
        "dDDG (kcal/mol)": "Uncertainty in computed relative free energy difference",
        "ref_mols": "a comma separated list of the fragments that inspired the design of the new molecule (codes as they appear in fragalysis - e.g. x0104_0,x0692_0)",
        "ref_pdb": "The name of the fragment (and corresponding Mpro fragment structure) with the best scoring hybrid docking pose",
        "original SMILES": "the original SMILES of the compound before any computation was carried out",
    }

    # Preprocess molecules
    tags_to_retain = {"DDG (kcal/mol)", "dDDG (kcal/mol)"}
    for oemol in track(molecules, "Preprocessing molecules..."):
        # Remove hydogrens
        oechem.OESuppressHydrogens(oemol, True)
        # Get original SMILES
        original_smiles = oechem.OEGetSDData(oemol, "SMILES")
        # Remove irrelevant SD tags
        for sdpair in oechem.OEGetSDDataPairs(oemol):
            tag = sdpair.GetTag()
            value = sdpair.GetValue()
            if tag not in tags_to_retain:
                oechem.OEDeleteSDData(oemol, tag)
        # Add required SD tags
        oechem.OESetSDData(oemol, "ref_mols", ref_mols)
        oechem.OESetSDData(oemol, "ref_pdb", ref_pdb)  # TODO: Upload corresponding PDBs
        oechem.OESetSDData(oemol, "original SMILES", original_smiles)

    # Add initial blank molecule (that includes distances)
    import copy
    from datetime import datetime

    # Find a molecule that includes distances, if present
    oemol = molecules[0].CreateCopy()
    # Add descriptions to each SD field
    for sdpair in oechem.OEGetSDDataPairs(oemol):
        tag = sdpair.GetTag()
        value = sdpair.GetValue()
        oechem.OESetSDData(oemol, tag, descriptions[tag])

    # Add other fields
    oemol.SetTitle("ver_1.2")
    oechem.OESetSDData(oemol, "ref_url", ref_url)
    oechem.OESetSDData(oemol, "submitter_name", submitter_name)
    oechem.OESetSDData(oemol, "submitter_email", "john.chodera@choderalab.org")
    oechem.OESetSDData(oemol, "submitter_institution", "MSKCC")
    oechem.OESetSDData(oemol, "generation_date", datetime.today().strftime("%Y-%m-%d"))
    oechem.OESetSDData(oemol, "method", method)
    molecules.insert(0, oemol)  # make it first molecule

    # Write sorted molecules
    with oechem.oemolostream(fragalysis_sdf_filename) as ofs:
        for oemol in track(molecules, description="Writing fragalysis SDF file..."):
            oechem.OEWriteMolecule(ofs, oemol)

    # Upload to fragalysis
    print("Uploading to fragalysis...")
    from fragalysis_api.xcextracter.computed_set_update import update_cset, REQ_URL

    upload_key = ""  # TODO add upload key
    update_set = "None"  # new upload
    update_set = "".join(submitter_name.split()) + "-" + "".join(method.split())
    cs_target_name = (
        "Mpro"  # name of the target in Fragalysis that the computed set is for
    )

    update_cset(
        REQ_URL,
        target_name=cs_target_name,
        fragalysis_sdf_filename,
        update_set=update_set,
        upload_key=upload_key,
        submit_choice=1,
        add=False,
    )


def gens_are_consistent(
    complex_phase,
    solvent_phase,
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
        complex_phase : ProjectPair
            The complex phase ProjectPair object to use to check for consistency
        solvent_phase : ProjectPair
            The solvent phase ProjectPair object to use to check for consistency
        ngens : int, optional, default=2
            The last `ngens` generations will be checked for consistency with the overall estimate
        nsigma : int, optional, default=3
            Number of standard errors of overall estimate to use for consistency check
    """
    # Collect free energy estimates for each GEN
    ngens = min(len(complex_phase.gens), len(solvent_phase.gens))
    gen_estimates = list()
    for gen in range(ngens):
        complex_delta_f = complex_phase.gens[gen].free_energy.delta_f
        solvent_delta_f = solvent_phase.gens[gen].free_energy.delta_f
        if (complex_delta_f is None) or (solvent_delta_f is None):
            continue
        binding_delta_f = complex_delta_f - solvent_delta_f
        gen_estimates.append(binding_delta_f)

    if len(gen_estimates) < ngens:
        # We don't have enough GENs
        return False

    # Flag inconsistent if any GEN estimate is more than nsigma stderrs away from overall estimate
    for gen_delta_f in gen_estimates[-ngens:]:
        overall_delta_f = (
            complex_phase.free_energy.delta_f - solvent_phase.free_energy.delta_f
        )
        delta_f = overall_delta_f - gen_delta_f
        # print(gen_delta_f, overall_delta_f, delta_f)
        # if abs(delta_f.point) > nsigma*delta_f.stderr:
        if abs(delta_f.point) > nsigma * gen_delta_f.stderr:
            return False

    return True


def generate_report(
    series: CompoundSeriesAnalysis,
    results_path: str,
    max_binding_free_energy: float = 0.0,
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
    # dictionary for target and reference molecules, with reliable and unreliable transformations
    mols = {
        "reliable": {"oemols": [], "refmols": []},
        "unreliable": {"oemols": [], "refmols": []},
    }

    # TODO : Iterate over compounds instead of transformations
    # Store optimal microstates for each compound, and representative snapshot paths for each microstate and compound in analysis
    for transformation in track(series.transformations, description="Reading ligands"):

        # Don't bother reading ligands with transformation free energies above max
        # since snapshots aren't generated for these
        if transformation.binding_free_energy.point >= max_binding_free_energy:
            continue

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

        mols["unreliable"]["refmols"].append(refmol)

        if filter_gen_consistency:
            if transformation.reliable_transformation:
                mols["reliable"]["refmols"].append(refmol)

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
        mols["unreliable"]["oemols"].append(oemol)

        if filter_gen_consistency:
            if transformation.reliable_transformation:
                mols["reliable"]["oemols"].append(oemol)

    logging.info(f"{len(mols['unreliable']['oemols'])} molecules read")

    # Sort ligands in order of most favorable transformations
    import numpy as np

    logging.info(f"Sorting molecules to prioritize most favorable transformations")
    sorted_indices = np.argsort(
        [
            float(oechem.OEGetSDData(oemol, "DDG (kcal/mol)"))
            for oemol in mols["unreliable"]["oemols"]
        ]
    )

    if filter_gen_consistency:
        sorted_indices_reliable = np.argsort(
            [
                float(oechem.OEGetSDData(oemol, "DDG (kcal/mol)"))
                for oemol in mols["reliable"]["oemols"]
            ]
        )

    # Filter based on threshold
    sorted_indices = [
        index
        for index in sorted_indices
        if (
            float(
                oechem.OEGetSDData(
                    mols["unreliable"]["oemols"][index], "DDG (kcal/mol)"
                )
            )
            < max_binding_free_energy
        )
    ]

    if filter_gen_consistency:
        sorted_indices_reliable = [
            index
            for index in sorted_indices_reliable
            if (
                float(
                    oechem.OEGetSDData(
                        mols["reliable"]["oemols"][index], "DDG (kcal/mol)"
                    )
                )
                < max_binding_free_energy
            )
        ]

    # Slice
    oemols = [mols["unreliable"]["oemols"][index] for index in sorted_indices]
    refmols = [mols["unreliable"]["refmols"][index] for index in sorted_indices]
    reliable_oemols = [
        mols["reliable"]["oemols"][index] for index in sorted_indices_reliable
    ]
    reliable_refmols = [
        mols["reliable"]["refmols"][index] for index in sorted_indices_reliable
    ]

    logging.info(
        f"{len(oemols)} molecules remain after filtering based on {max_binding_free_energy} threshold"
    )

    # Write sorted molecules
    for filename in [
        "transformations-final-ligands.sdf",
        "transformations-final-ligands.csv",
        "transformations-final-ligands.mol2",
    ]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for oemol in track(oemols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, oemol)

    if filter_gen_consistency:
        for filename in [
            "reliable-transformations-final-ligands.sdf",
            "reliable-transformations-final-ligands.csv",
            "reliable-transformations-final-ligands.mol2",
        ]:
            with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
                for oemol in track(reliable_oemols, description=f"Writing {filename}"):
                    oechem.OEWriteMolecule(ofs, oemol)

    # Write PDF report
    write_pdf_report(
        oemols,
        os.path.join(results_path, "transformations-final-ligands.pdf"),
        series.metadata.name,
    )

    if filter_gen_consistency:
        write_pdf_report(
            reliable_oemols,
            os.path.join(results_path, "reliable-transformations-final-ligands.pdf"),
            series.metadata.name,
        )

    # Write reference molecules
    for filename in [
        "transformations-initial-ligands.sdf",
        "transformations-initial-ligands.mol2",
    ]:
        with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
            for refmol in track(refmols, description=f"Writing {filename}"):
                oechem.OEWriteMolecule(ofs, refmol)

    if filter_gen_consistency:
        for filename in [
            "reliable-transformations-initial-ligands.sdf",
            "reliable-transformations-initial-ligands.mol2",
        ]:
            with oechem.oemolostream(os.path.join(results_path, filename)) as ofs:
                for refmol in track(
                    reliable_refmols, description=f"Writing {filename}"
                ):
                    oechem.OEWriteMolecule(ofs, refmol)

    if consolidate_protein_snapshots:
        consolidate_protein_snapshots_into_pdb(oemols, results_path)
        if filter_gen_consistency:
            consolidate_protein_snapshots_into_pdb(
                reliable_oemols,
                results_path,
                pdb_filename="reliable-transformations-final-proteins.pdb",
            )


from openeye import oechem


def consolidate_protein_snapshots_into_pdb(
    oemols: List[oechem.OEMol],
    results_path: str,
    pdb_filename: Optional[str] = "transformations-final-proteins.pdb",
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
        return  # DEBUG
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
