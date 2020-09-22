from functools import partial
from hashlib import sha256
import multiprocessing
import logging
import os
from typing import List

from ...schema import Microstate


def generate_molecule_images(
    microstates: List[Microstate],
    path: str,
) -> None:
    os.makedirs(path, exist_ok=True)
    render_molecule_partial = partial(render_molecule, path=path)
    smiless = [microstate.smiles for microstate in microstates]
    with multiprocessing.Pool() as pool:
        for _ in pool.imap_unordered(render_molecule_partial, smiless):
            pass


def get_image_filename(smiles: str):
    return sha256(smiles.encode()).hexdigest()


def render_molecule(
    smiles: str,
    path: str,
    width: int = 320,
    height: int = 240,
    file_format: str = "svg",
    clearbackground: bool = False,
    force_regenerate: bool = False,
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

    output_name = get_image_filename(smiles)
    output_path = os.path.join(path, os.extsep.join([output_name, file_format]))

    if not force_regenerate and os.path.exists(output_path):
        logging.info("Skipping already-rendered molecule: %s", smiles)
        return

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
    oedepict.OERenderMolecule(output_path, disp, clearbackground)
