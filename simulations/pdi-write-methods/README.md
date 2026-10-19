# GYSELA parallel-write benchmark

Benchmark of the ways PDI can write the 6D distribution function `fdistribu`
(`species × tor1 × tor2 × tor3 × vpar × mu`, double precision) on a parallel file system
(developed on Adastra / Lustre). All strategies write the same payload; only the file layout
and the PDI plugin change.

| `method`          | PDI plugin    | Layout                                                       |
|-------------------|---------------|--------------------------------------------------------------|
| `shared_netcdf`   | `decl_netcdf` | all ranks → one NetCDF-4 file                                |
| `per_rank_netcdf` | `decl_netcdf` | each rank → its own NetCDF-4 file                            |
| `per_rank_hdf5`   | `decl_hdf5`   | each rank → its own HDF5 file                                |
| `shared_hdf5`     | `decl_hdf5`   | all ranks → one HDF5 file                                    |
| `subfiling_hdf5`  | `decl_hdf5`   | all ranks → one logical HDF5 file stored as several subfiles |

## Application

`src/pdi-write-methods/gys_write_methods.cpp` initialises a Maxwellian and triggers one `write_fdistribu` event.
Ranks start together (barrier); the reported `Time write:` is the slowest rank and includes
`sync()`. Subfiling needs `MPI_THREAD_MULTIPLE` (on Cray MPICH also
`MPICH_MAX_THREAD_SAFETY=multiple`; the IOPS configs set it).

## Build

## Benchmarks with IOPS

[IOPS](https://github.com/io-performance-suite/iops) generates the inputs, sets the Lustre
striping, runs the application with `srun` and parses the bandwidth.

| File                       | Question                              | Sweep                                                                              |
|----------------------------|---------------------------------------|------------------------------------------------------------------------------------|
| `iops_config_default.yaml` | Which write method is fastest?        | 5 methods × stripe {1, 8, all}, 4 nodes, ~128 GiB                                  |
| `gys_scaling_bench.yaml`   | Where does each method stop scaling?  | per-rank NetCDF, shared HDF5, subfiling (1 or 4 IOC/node), 4 / 8 / 16 nodes, ~256 GiB |

A run fails if the bytes written are not the expected payload (at most 2 % above), if the file
count does not match the method.

```bash
iops check --machine adastra simulations/pdi-write-methods/iops_config_default.yaml
iops run   --machine adastra simulations/pdi-write-methods/iops_config_default.yaml
```

## Plotting the results

Once a benchmark has finished, draw the plots from its results CSV:

```bash
python processing/pdi-write-methods/plot_summary.py output/results_write_methods.csv
python processing/pdi-write-methods/plot_summary.py output/results_scaling_256GiB.csv
```

The figures are written next to the CSV (`-o DIR` to change, `--format png pdf`, `--theme dark`).
