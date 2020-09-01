import os
from typing import List
from ..core import RunDetails


def save_molecule_images(
    runs: List[RunDetails], path: str, file_format: str = "svg"
) -> None:
    os.makedirs(path, exist_ok=True)
    for run in runs:
        render_molecule(
            smiles=run.start_smiles,
            filename=os.path.join(
                path, os.extsep.join([f"RUN{run.run_id()}", file_format])
            ),
        )


def render_molecule(
    smiles: str,
    filename: str,
    width: int = 320,
    height: int = 240,
    clearbackground: bool = False,
) -> None:
    """
    Render the molecule (from SMILES) to an image
    Parameters
    ----------
    smiles : str
        The SMILES string
    filename : str
        Output filename, with image format (pdf, png, jpg) detected from filename
    width : int, optional, default=320
        Default image width
    height : int, optional, default=240
        Default image height
    clearbackground : bool, optional, default=False
        For PNG, whether background should be clear
    """
    # Import the openeye toolkit
    from openeye import oechem, oedepict

    # Generate OpenEye OEMol object from SMILES
    # see https://docs.eyesopen.com/toolkits/python/oechemtk/molctordtor.html?highlight=smiles#construction-from-smiles
    mol = oechem.OEGraphMol()

    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"Failed to convert SMILES string to molecule: {smiles}")

    # Configure options (lots more are available)
    # see https://docs.eyesopen.com/toolkits/python/depicttk/OEDepictClasses/OE2DMolDisplayOptions.html
    opts = oedepict.OE2DMolDisplayOptions()
    opts.SetWidth(width)
    opts.SetHeight(height)

    # Render image
    oedepict.OEPrepareDepiction(mol)
    disp = oedepict.OE2DMolDisplay(mol, opts)
    oedepict.OERenderMolecule(filename, disp, clearbackground)
