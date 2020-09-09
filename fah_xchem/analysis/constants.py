from simtk.openmm import unit
from openmmtools.constants import kB

# TODO: should be a configuration parameter
KT_KCALMOL = kB * 300 * unit.kelvin / unit.kilocalories_per_mole
