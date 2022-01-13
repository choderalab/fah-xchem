import pandas as pd

from ..schema import DataPath, WorkPair
from .exceptions import DataValidationError


def _is_header_line(line: str) -> bool:
    """
    Determine if the specified line is a globals.csv header line

    Parameters
    ----------
    line : str
        The line to evaluate

    Returns
    -------
    is_header_line : bool
        If True, `line` is a header line

    """
    return "kT" in line


def _get_last_header_line(path: str) -> int:
    """
    Return the line index of the last occurrence of the globals.csv header
    in order to filter out erroneously repeated headers

    Parameters
    ----------
    path : str
        The path to the globals.csv file

    Returns
    -------
    index : int
        The index of the last header line
    """
    with open(path, "r") as f:
        lines = f.readlines()
    header_lines = [i for i, line in enumerate(lines) if _is_header_line(line)]
    if not header_lines:
        raise ValueError(f"Missing header in {path}")
    return header_lines[-1]


def _get_num_steps(df: pd.DataFrame) -> int:
    """
    Get the number of steps described in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing `Step` column

    Returns
    -------
    n_steps : int
        The number of steps described in the dataframe

    """
    if df.empty:
        raise ValueError("Empty dataframe")
    step = df["Step"].astype(int)
    return int(step.iloc[-1] - step.iloc[0])


def extract_work_pair(path: DataPath) -> WorkPair:
    """
    Extract forward and reverse protocol work from a globals.csv file

    Parameters
    ----------
    path : DataPath
        Path to `globals.csv`

    Returns
    -------
    work : WorkPair
        The forward and reverse dimensionless protocol work

    """
    
    # TODO: magic numbers
    NUM_WORKS_EXPECTED = [41, 401]
    NUM_STEPS_EXPECTED = 1000000

    # TODO: Because of a known bug in core22 0.0.11,
    # globals.csv can have duplicate headers or duplicate records
    # if the core is paused and resumed.
    # https://github.com/FoldingAtHome/openmm-core/issues/281

    # Start with the last header entry (due to aforementioned bug)
    header_line_number = _get_last_header_line(path.path)
    df = pd.read_csv(path.path, header=header_line_number)

    # Drop any dupliates we many encounter (due to aforementioned bug)
    df = df.drop_duplicates()

    # Extract the thermal energy in openmm units (kJ/mol, but could change)
    kT = df["kT"].astype(float)[0]

    # Extract the cumulative protocol work in openmm units (kJ/mol)
    # and convert it to dimensionless work
    protocol_work = df["protocol_work"].astype(float).values
    protocol_work_nodims = protocol_work / kT

    # Extract the new potential energy in openmm units (kJ/mol)
    # and convert it to dimensionless energy
    # Enew = df["Enew"].astype(float).values
    # Enew_nodims = Enew / kT

    # Check to make sure we don't have an incorrect number of work values
    # TODO: Diagnose why this happens and file an issue in core
    # https://github.com/FoldingAtHome/openmm-core/issues
    if len(protocol_work_nodims) not in NUM_WORKS_EXPECTED:
        raise DataValidationError(
            f"Expected {NUM_WORKS_EXPECTED} work values, "
            f"but found {len(protocol_work_nodims)}",
            path=path,
        )

    # Determine number of work values per phase
    nworks_per_phase = int((len(protocol_work_nodims)-1)/4)
    
    # Check to make sure we don't have an incorrect number of steps
    # TODO: Diagnose why this happens and file an issue in core
    # https://github.com/FoldingAtHome/openmm-core/issues
    num_steps = _get_num_steps(df)
    if num_steps != NUM_STEPS_EXPECTED:
        raise DataValidationError(
            f"Expected {NUM_STEPS_EXPECTED} steps, but found {num_steps}", path=path
        )

    # TODO: magic numbers
    try:
        return WorkPair(
            clone=path.clone,
            forward=protocol_work_nodims[2*nworks_per_phase] - protocol_work_nodims[1*nworks_per_phase],
            reverse=protocol_work_nodims[4*nworks_per_phase] - protocol_work_nodims[3*nworks_per_phase],
            # forward_final_potential=Enew_nodims[20],
            # reverse_final_potential=Enew_nodims[40],
        )
    except KeyError as e:
        raise DataValidationError(
            f"Tried to index into dataframe at row {e}, "
            f"but dataframe has {len(df)} rows",
            path=path,
        )
