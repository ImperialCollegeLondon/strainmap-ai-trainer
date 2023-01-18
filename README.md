# StrainMap AI trainer

Training code for the AI used by StrainMap.

## Installation

After downloading or cloing this repository, navigate to the root directory and do:

### For development purposes

```bash
conda env create --file environment.yml
```

which will install the `pytest` testing framework and install `tensorflow` compiled for
using CPUs.

Then activate the environment with:

### For production in an appropriate GPU-capable system

```bash
conda env create --file environment_gpu.yml
```

which will not install `pytest` and `tensorflow` will be compiled with GPU support.

## Usage

First activate the environment:

```bash
source activate strainmap_ai
```

Then, run the trainig mmodule:

```bash
python -m strainmap_ai NETCDF_FILES --model_path TRAINED_MODEL --test PATIENT1,PATIENT2
```

Where:
- NETCDF_FILES: Parent directory where the `_train.nc` files (as produced by StrainMap)
  are located
- TRAINED_MODEL (optional): Location where to save the trained model.
- PATIENT1,PATIENT2,... (optional): Comma separated list of patients initials (must match the
  initials used in the NetCDF files) that will be used for testing. All the others will
  be used for trainnig.

### At Imperial's HPC

To run the tool at Imperial's HPC, edit the `train_strainmap.pbs` file so it runs the
calculation as desired (see previous section) and then:

```bash
qsub train_strainmap.pbs
```

You can check the status of your calculation running `qstat` without any arguments.