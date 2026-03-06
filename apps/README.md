# GYSELA Mini App

A minimal application demonstrating GYSELA I/O operations and testing the cpu performance scaling for 5D particle distribution functions.

## Overview

This mini application:

- Initialises a 5D particle distribution function (species × toroidal coordinates × velocity space)
- Writes the distribution function and mesh coordinates to HDF5 files
- Computes fluid moments (density, mean velocity, temperature) via C++ integration or in-situ Python computation
- Measures and saves CPU timing statistics

## Usage

```bash
mpirun -n <nprocs> ./gys_io [config.yaml] [pdi_config.yml]
```

- `config.yaml`: Input configuration file (default: uses built-in defaults)
- `pdi_config.yml`: PDI configuration file (default: uses `pdi_default.yml.hpp`)

### Example

```bash
mpirun -n 4 ./gys_io gys_io.yaml
```
Do not forget to set the the `PYTHONPATH` if you are using PyCall:

```bash
export PYTHONPATH=/home/PK282018/develop/gysela-mini-app_io/python:$PYTHONPATH
```

## Configuration

Edit `gys_io.yaml` to configure:

- **Mesh**: Grid sizes and ranges for toroidal coordinates (Tor1, Tor2, Tor3) and velocity space (Vpar, Mu)
- **Species**: Number of species, charges, masses
- **Application version**: `"gpu2cpu"`, `"mpi_transpose"`, or `"in-situ-diagnostic"` (the latter enables Python-based fluid moments computation)

## Output Files

- **`fdistribu_5D_output.h5`**: Distribution function and mesh coordinates
- **`cpu_time_stats.h5`**: CPU timing statistics (initialisation, transpose, GPU↔CPU transfer, I/O)
- **`fluid_moments.h5`**: Fluid moments (density, mean velocity, temperature)
