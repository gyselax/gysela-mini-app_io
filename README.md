# Gysela Mini App I/O

A minimal application demonstrating GYSELA I/O operations and testing the CPU performance scaling for 5D particle distribution functions.

This repository contains mini apps that allow easy coupling with GyselaX++.

## Installing

```bash
git clone git@github.com:gyselax/gysela-mini-app_io.git
cd gysela-mini-app_io
git submodule update --init --recursive
```

## Sourcing the environment

```bash
source external/gyselalibxx/toolchains/<MACHINE>/environment.sh
```

For more details see [Gyselalib++ environment toolchains](https://gyselax.github.io/gyselalibxx/toolchains/index.html#environment-setup).

## Building

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=external/gyselalibxx/toolchains/<MACHINE>/toolchain.cmake
cmake --build build -j 4
```
if you want to use Python insitu-diagnostics set additionally the `PYTHONPATH`:

```bash
export PYTHONPATH=/home/PK282018/develop/gysela-mini-app_io/python:$PYTHONPATH
```

## Running

```bash
mpirun -n <nprocs> ./build/apps/gys_io <config.yaml> <pdi_config.yaml>
```

### Example

```bash
mpirun -n 4 ./build/apps/gys_io apps/gys_io.yaml apps/pdi_default.yaml
```
Note that `nprocs>1` works currently only with the version `"mpi_transpose"` switched on in the `apps/gys_io.yaml` yaml file. 

## Configuration

Edit `apps/gys_io.yaml` to configure:

- **Mesh**: Grid sizes and ranges for toroidal coordinates (Tor1, Tor2, Tor3) and velocity space (Vpar, Mu)
- **Species**: Number of species, charges, masses
- **Application version**: `"mpi_transpose"` or `"in-situ-diagnostic"`

## Output Files

- **`fdistribu_5D_output.h5`**: Distribution function and mesh coordinates
- **`cpu_time.h5`**: CPU timing statistics (initialisation, transpose, GPU↔CPU transfer, I/O)
- **`fluid_moments.h5`**: Fluid moments (density, mean velocity, temperature)

## Overview

This mini application:

- Initialises a 5D particle distribution function (species × toroidal coordinates × velocity space)
- Writes the distribution function and mesh coordinates to HDF5 files
- Computes fluid moments (density, mean velocity, temperature) via C++ integration or in-situ Python computation
- Measures and saves CPU timing statistics
