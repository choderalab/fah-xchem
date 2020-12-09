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
    conda env create -f environment.yml
    ```

    If the above process is slow, we recommend using [mamba](https://github.com/mamba-org/mamba.git) to speed up installation:

    ```sh
    mamba env create -f environment.yml
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
        --fragalysis-config fragalysis_config.json
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

#### Upload to Fragalysis

To upload sprint results to [Fragalysis](https://fragalysis.diamond.ac.uk/viewer/react/landing) a JSON config file may be supplied. For example,

`fragalysis_config.json`
```json
{
        "run": true,
        "ligands_filename": "reliable-transformations-final-ligands.sdf",
        "proteins_filename": "reliable-transformations-final-proteins.pdb",
        "fragalysis_sdf_filename": "compound-set_foldingathome-sprint-X.sdf",
        "ref_url": "https://url-link",
        "ref_mols": "x00000",
        "ref_pdb": "x00000",
        "target_name": "protein-target",
        "submitter_name": "Folding@home",
        "submitter_email": "first.last@email.org",
        "submitter_institution": "institution-name",
        "method": "Sprint X",
        "upload_key": "upload-key",
        "new_upload": true 
}
```

The JSON file is passed on the command line using the `--fragalysis-config` option.

Description of the JSON parameters:

* `run`: specify whehter to run the Fragalysis upload. If set to `false` the results will not be uploaded (even if the JSON is supplied via the `--fragalysis-config` option).
* `ligands_filename`: the name of the SDF file to upload to Fragalysis.
* `proteins_filename`: the name of the PDB file to upload to Fragalysis - **not implemented yet**.
* `fragalysis_sdf_filename`: the name to use for the SDF Fragalysis upload. This will be a copy of `ligands_filename` but must be in the form `compound-set_name.sdf`.
* `ref_url`: the url to the post that describes the work e.g. for [Sprint 5](https://discuss.postera.ai/t/folding-home-sprint-5/2423).
* `ref_mol`: a comma separated list of the fragments that inspired the design of the new molecule (codes as they appear in fragalysis - e.g. x0104_0,x0692_0).
* `ref_pdb`: 1) the file path of the pdb file in the uploaded zip file or 2) the code to the fragment pdb from fragalysis that should be used (e.g. x0692_0).
* `target_name`: the name of the target protein.
* `submitter_name`: the name of the submitter.
* `submitter_email`: the email address of the submitter.
* `submitter_institution`: the name of the institution that the submitter is associated with.
* `method`: the method by which the results were obtained (e.g. Sprint 5).
* `upload_key`: the unique upload key used to upload to Fragalysis.
* `new_upload`: specifies whether to upload a new set (`true`) or to update an existing set (`false`).

For more information see this forum [post](https://discuss.postera.ai/t/providing-computed-poses-for-others-to-look-at/1155/8).

#### Server-specific configuration

Paths to Folding@home project and data directories are passed on the command line. See usage examples above.

### Development setup

#### Conda

This project uses [conda](https://github.com/conda/conda) to manage the environment. To set up a conda environment named `fah-xchem` with the required dependencies, create the conda environment as described above. To install `fah-xchem` as `dev` run:

```sh
pip install -e .
```

#### Running tests locally

``` sh
pytest
```

#### Formatting

Code formatting with [black](https://github.com/psf/black) is enforced via a CI check. To install `black` with `conda`, use

``` sh
conda install black
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
