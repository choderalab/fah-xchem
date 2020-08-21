import pandas as pd
from .core import ResultPath, Work


def _is_header_line(line: str) -> bool:
    return "kT" in line


def _get_last_header_line(path: str) -> int:
    with open(path, "r") as f:
        lines = f.readlines()
    header_lines = [i for i, line in enumerate(lines) if _is_header_line(line)]
    if not header_lines:
        raise ValueError(f"Missing header in {path}")
    return header_lines[-1]


def _get_num_steps(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("Empty dataframe")
    step = df["Step"].astype(int)
    return step.iloc[-1] - step.iloc[0]


def extract_work(path: ResultPath) -> Work:

    NUM_WORKS_EXPECTED = 41
    NUM_STEPS_EXPECTED = 1000000

    header_line_number = _get_last_header_line(path.path)
    df = pd.read_csv(path.path, header=header_line_number)

    # TODO: explanation for duplicates?
    df = df.drop_duplicates()

    kT = df["kT"].astype(float)[0]

    protocol_work = df["protocol_work"].astype(float).values
    protocol_work_nodims = protocol_work / kT

    Enew = df["Enew"].astype(float).values
    Enew_nodims = Enew / kT

    if len(protocol_work_nodims) != NUM_WORKS_EXPECTED:
        raise ValueError(
            f"Expected {NUM_WORKS_EXPECTED} work values, "
            f"but found {len(protocol_work_nodims)}"
        )

    num_steps = _get_num_steps(df)
    if num_steps != NUM_STEPS_EXPECTED:
        raise ValueError(f"Expected {NUM_STEPS_EXPECTED} steps, but found {num_steps}")

    # TODO: magic numbers
    try:
        return Work(
            path=path,
            forward_work=protocol_work_nodims[20] - protocol_work_nodims[10],
            reverse_work=protocol_work_nodims[40] - protocol_work_nodims[30],
            forward_final_potential=Enew_nodims[20],
            reverse_final_potential=Enew_nodims[40],
        )
    except KeyError as e:
        raise ValueError(
            f"Tried to index into dataframe at row {e}, "
            f"but dataframe has {len(df)} rows"
        )
