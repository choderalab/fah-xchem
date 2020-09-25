fah-xchem
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/choderalab/fah-xchem/workflows/CI/badge.svg)](https://github.com/choderalab/fah-xchem/actions?query=branch%3Amaster+workflow%3ACI)
[![codecov](https://codecov.io/gh/choderalab/fah-xchem/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/fah-xchem/branch/master)

Tools and infrastructure for automated compound discovery using Folding@home.

### Installation

1. Clone the repository and `cd` into repo root:

    ``` sh
    git clone https://github.com/choderalab/fah-xchem.git
    cd fah-xchem
    ```

2. Create a `conda` environment with the required dependencies:

    ``` sh
    conda env create -n fah-xchem
    ```

3. Install `fah-xchem` in the environment using `pip`:

    ``` sh
    pip install .
    ```

### Example usage

Run transformation and compound free energy analysis, producing `results/analysis.json`:

``` sh
fah-xchem run-analysis
        --compound-series-file compound-series.json \
        --config-file config.json \
        --fah-projects-dir /path/to/projects/ \
        --fah-data-dir /path/to/data/SVR314342810/ \
        --output-dir results \
        --log INFO \
        --num-procs 8
```


Generate representative snapshots, plots, PDF report, and static site HTML in `results` directory:
``` sh
fah-xchem generate-artifacts \
        --compound-series-analysis-file results/analysis.json \
        --config-file config.json \
        --fah-projects-dir /path/to/projects/ \
        --fah-data-dir /path/to/data/SVR314342810/ \
        --output-dir results \
        --base-url https://my-bucket.s3.amazonaws.com/site/prefix/ \
        --cache-dir results/cache/ \
        --num-procs 8
```


### Unit conventions

Energies are represented in configuration and internally in units of `k T`, except when otherwise indicated. For energies in kilocalories per mole, the function or variable name should be suffixed with `_kcal`.

### Configuration

#### Compound series
The compound series is specified as JSON with schema given by the `CompoundSeriesAnalysis` model (see [fah_xchem.schema](fah_xchem/schema.py).

The JSON file is passed on the command line using the `--compound-series-file` option

#### Analysis configuration
Some analysis options can be configured in a separate JSON file with schema given by the `AnalysisConfig` model. For example,

`config.json`
``` json
{
    "min_num_work_values": 10,
    "max_binding_free_energy": 0
}
```

The JSON file is passed on the command line using the `--config-file` option.

#### Server-specific configuration

Paths to Folding@home project and data directories are passed on the command line. See usage examples above.

### Development setup

#### Conda

This project uses [conda](https://github.com/conda/conda) to manage the environment. To set up a conda environment named `fah-xchem` with the required dependencies, run

``` sh
conda env create -n fah-xchem
```

#### Running tests locally

``` sh
pytest
```

#### Formatting

Code formatting with [black](https://github.com/psf/black) is enforced via a CI check. To install `black` with `conda`, use

``` sh
conda env create -n fah-xchem
```

#### Building documentation

``` sh
cd docs
make html
```


### Copyright

Copyright (c) 2020, Chodera Lab


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
