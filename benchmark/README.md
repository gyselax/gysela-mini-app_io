# `gys_io` I/O Benchmark — IOPS Configuration

This directory contains the [IOPS](https://iops-benchmark.com) configuration file used to benchmark the parallel I/O performance of `gys_io` on the Gysela codebase. It systematically sweeps over I/O strategies, grid sizes, and storage parameters to evaluate write throughput under realistic conditions.

---

## Overview

The benchmark exercises three HDF5-based I/O strategies for writing checkpoint data from `gys_io`:

| Strategy | Description |
|---|---|
| `ssf` | Single Shared File — all MPI ranks write to one collective HDF5 file |
| `subfiling` | HDF5 Subfiling VFD — collective write split across *N* subfiles |
| `fpp` | File Per Process — each MPI rank writes its own independent HDF5 file |

Each run produces two timed metrics extracted from the job output:

- `time_write` — time spent in write operations
- `time_total` — total execution time of `gys_io`

Results are aggregated into a CSV file (`results_gys_io.csv`) at the end of the sweep.

---

## Requirements

- **IOPS** installed and available in the python environment
- **`gys_io`** binary compiled and available in `$PATH`
- Environment variable `GYS_IO_OUTPUT_PATH` set to the root directory where output HDF5 files will be written (see [Environment Variables](#environment-variables))

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GYS_IO_OUTPUT_PATH` | Yes | Root output path for HDF5 files. OST-specific subdirectories (`ost_<N>`) are used under this path when `ost_count != 0`. |
| `gys_io` | Yes | Path to the `gys_io` executable, must be in `$PATH`. |
| `H5FD_SUBFILING_STRIPE_SIZE` | Only for `subfiling` | Stripe size in bytes for the Subfiling VFD. Set automatically by IOPS via `mpirun -x`. |

---

## Install IOPS

```bash
pip install iops-benchmark
```

## Running the Benchmark

```bash
iops run iops_gys.yaml
```

To target the `plafrim` machine profile (Slurm, 3 repetitions, `bora` partition):
(Fell free to adjust or add more machine profiles as needed to overide execution settings for different environments)

```bash
iops run iops_gys.yaml --machine plafrim
```

IOPS will:
1. Expand the variable sweep into a full parameter matrix
2. Generate per-run `gys_io.yaml` and `pdi_config.yaml` config files from the embedded templates
3. Submit or execute jobs and collect `time_write` / `time_total` from stdout
4. Cache results in `iops_gys.db` and write the final CSV

---

## Key Parameters

### Sweep Variables

Those actual variables values are defined for presentation purposes, adjust or add new one as needed for your testing.

| Variable | Values swept | Description |
|---|---|---|
| `nodes` | `[8]` | Number of compute nodes |
| `tasks_per_nodes` | `[4]` | MPI ranks per node |
| `file_strategy` | `ssf`, `subfiling`, `fpp` | I/O strategy (see table above) |
| `subfile_number` | `1`, `2` | Number of subfiles *(subfiling only)* |
| `subfile_stripe_size_bytes` | `536870912` (512 MB) | Stripe size per subfile *(subfiling only)* |
| `ost_count` | `1`, `4`, `8` | Number of Lustre OSTs; selects the output subdirectory |
| `grid_index` | `0`, `1` | Selects the grid resolution preset (see below) |

### Grid Presets

Two grid sizes are available, selected via `grid_index`:

| Dimension | `grid_index=0` (small) | `grid_index=1` (medium) |
|---|---|---|
| `tor1` | 15 | 125 |
| `tor2` | 32 | 128 |
| `tor3` | 32 | 64 |
| `vpar` | 15 | 63 |
| `mu` | 9 | 63 |

### Conditional Variables

`subfile_number` and `subfile_stripe_size_bytes` are only included in the sweep when `file_strategy == 'subfiling'`; they default to `0` otherwise. Similarly, `pfs_path` resolves to an `ost_<N>` subdirectory when `ost_count != 0`, and falls back to `GYS_IO_OUTPUT_PATH` directly.

---

## Output Files

Each run execution directory contains:

| File | Description |
|---|---|
| `gys_io.yaml` | Generated `gys_io` configuration (mesh, species, filenames) |
| `pdi_config.yaml` | Generated PDI configuration (I/O strategy, HDF5 datasets, events) |
| `gys_io.out` | stdout/stderr of the `gys_io` run; parsed for timing metrics |
| `summary.txt` | Summary output written by `gys_io` |

On the PFS (`pfs_path`):

| File | Description |
|---|---|
| `fdistribu_5D_output[_<rank>].h5` | Distribution function (`fpp` appends `_<mpi_rank>` to filename) |
| `fluid_moments.h5` | Fluid moments (density, mean velocity, temperature) |
| `cpu_time.h5` | Timing table written by `gys_io` |

---

## Machine Profile: `plafrim`

The `plafrim` profile overrides the default local executor with a Slurm submission targeting the `bora` partition. Key Slurm settings:

- `--exclusive` — dedicated nodes, no sharing
- `--constraint='bora'`
- `--exclude=bora042,bora023,bora019` — known problematic nodes excluded
- Wall time: 30 minutes per job
- Repetitions: 3 (vs. 1 for local execution)