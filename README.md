# StrainMap AI trainer

Training code for the AI used by StrainMap.

For development purposes, use:

```bash
conda env create --file environment.yml
```

which will install the `pytest` testing framework and install `tensorflow` compiled for
using CPUs.

For production in an appropriate GPU-capable system, use:

```bash
conda env create --file environment_gpu.yml
```

which will not install `pytest` and `tensorflow` will be compiled with GPU support.