from simtk.openmm import unit
from openmmtools.constants import kB
import numpy as np

# TODO: should be a configuration parameter
KT_KCALMOL = kB * 300 * unit.kelvin / unit.kilocalories_per_mole
KCALMOL_KT = 1.0 / KT_KCALMOL
KT_PIC50 = np.log10(np.e)
