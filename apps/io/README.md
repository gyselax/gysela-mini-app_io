# GYSELA I/O Mini App

A minimal application demonstrating GYSELA I/O operations and testing the cpu performance scaling for 5D particle distribution functions.

## Overview

This mini application:

- Initialises a 5D particle distribution function (species × toroidal coordinates × velocity space)
- Writes the distribution function and mesh coordinates to HDF5 files
- Computes fluid moments (density, mean velocity, temperature) via C++ integration or in-situ Python computation
- Measures and saves CPU timing statistics

### Installation

For CPU use cases :
```bash
./toolchains/cpu.spack.mini_app_io_env/prepare.sh
```
Then insert the correct <MACHINE> toolchain path in the `gysela-mini-app_io/apps/io/activate_deisa_spack_env.sh` line 4.

### Persee XEON

If you run on persee, the environment is already available. You have nothing to change.

## Usage

### Basic run

```bash
./apps/io/deisa-dask_launch_script.sh <nsimu_procs> <nworker_proc> [pdi_config.yml] [analytics_script.py]
```

- `nsimu_procs`: number of MPI ranks for the simulation
- `nworker_proc`: number of Dask workers to use for the analytics
- `pdi_config.yml`: PDI configuration file (default: uses `pdi_deisa.yaml`)
- `analytics_script.py`: the analytics script to launch (either `analytics.py` if you chose `pdi_deisa.yaml`, or `optimised_analytics.py` if you chose `optimised_pdi_deisa.yaml`)


### Sequential Run

```bash
source toolcahins/<machine>/[prepare.sh | environment.sh]
mpirun -n <nprocs> ./build/apps/io/gys_io [config.yaml] seq_pdi.yaml
```

### Run with OAR

```bash
./apps/io/oar_deisa-dask_launch_script.sh <nsimu_procs> <nworker_proc>
```

`.gys_env/` is created by `./installer.sh` at the repository root.

## Configuration

Edit `gys_io.yaml` to configure:

- **Mesh**: Grid sizes and ranges for toroidal coordinates (Tor1, Tor2, Tor3) and velocity space (Vpar, Mu)
- **Species**: Number of species, charges, masses
- **Application version**: `"gpu2cpu"`, `"mpi_transpose"`, or `"in-situ-diagnostic"` (the latter enables Python-based fluid moments computation)

## Output Files

- **`fdistribu_5D_output.h5`**: Distribution function and mesh coordinates
- **`cpu_time_stats.h5`**: CPU timing statistics (initialisation, transpose, GPU↔CPU transfer, I/O)
- **`fluid_moments.h5`**: Fluid moments (density, mean velocity, temperature)
