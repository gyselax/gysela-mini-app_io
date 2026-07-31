# GYSELA Compression Mini App

Benchmark pipeline for evaluating restart-file compression in a 2D2V Vlasov--Poisson mini-application. The workflow compares an uninterrupted baseline against a segmented simulation that periodically compresses and decompresses the distribution function before restarting. Default case: Landau damping.

## Files

| File | Role |
| --- | --- |
| `gys_compress.cpp` | C++ mini-app: cold start and restart via PDI, writes `GYSELALIBXX_*.h5`. |
| `launch_benchmark.py` | Orchestrates the full benchmark: Dask lifecycle, in-situ diagnostics, PCA compression at every restart point, `compression_events.yaml`. |
| `evaluate_compression.py` | Two subcommands: `diagnostics` plots `diagnostics.csv` files; `online-networks` reloads saved `params_iterXXXXX_rankXXX.npz` payloads and plots the global multi-rank reconstruction / each rank's local network against the local data it was trained on. |
| `PCA.py` | PCA compressor for `fdistribu[sp, x, y, vx, vy]`. |
| `params_landau_damping.yaml` | Landau-damping input for `launch_benchmark.py`. |
| `params_two_stream.yaml` | Two-stream instability input. |
| `pdi_out.yaml` | Plain PDI config (HDF5 only). Use for standalone runs without diagnostics. |
| `pdi_out_diags.yaml` | PDI config with deisa Bridge. Used by `launch_benchmark.py` and `deisa-dask_launch_script.sh`. |
| `deisa-dask_launch_script.sh` | Shell script for manual single-segment runs with in-situ diagnostics. |

## Run directory layout

```text
compression_run_YYYYMMDD_HHMMSS/
├── branch_baseline/
│   ├── diagnostics.csv            # in-situ conserved-variable diagnostics
│   └── GYSELALIBXX_initstate.h5
├── branch_compressed/
│   ├── diagnostics.csv            # appended across restart segments
│   └── GYSELALIBXX_*.h5          # snapshots at compression boundaries only
├── periodic_restarts/
│   ├── restart_iter_XXXXX_approx.h5
│   └── restart_iter_XXXXX_compressed.npz
├── config_baseline.yaml
└── compression_events.yaml
```

The baseline suppresses all intermediate HDF5 snapshots; conserved variables come entirely from in-situ diagnostics. The compressed branch writes HDF5 only at `compression_period` boundaries.

## Environment

Follow the deisa/Dask installation instructions in [`apps/io/README.md`](../io/README.md#installation) and set up `apps/io/activate_deisa_spack_env.sh` for your machine.

## Running the benchmark

```bash
python apps/compression/launch_benchmark.py [run_dir] [options]
```

`launch_benchmark.py` handles everything: loads the deisa environment, starts a Dask cluster before each simulation segment, runs in-situ diagnostics in parallel with the simulation, and shuts Dask down after each segment.

| Option | Description |
| --- | --- |
| `--overwrite` | Reuse an existing non-empty run directory. |
| `--dask-workers N` | Number of Dask workers per segment (default: 1). |
| `--keep-payloads` | Keep `restart_iter_XXXXX_compressed.npz` files. |
| `--keep-restart-approximations` | Keep `restart_iter_XXXXX_approx.h5` files. |
| `--keep-segment-configs` | Keep temporary segment YAML configs. |
| `--keep-pdi-copy` | Keep the copy of `pdi_out_diags.yaml` in the run directory. |

Adjust `EXEC_CMD` in `launch_benchmark.py` if the MPI rank count or executable path must change.

## Running the mini-app directly

Without diagnostics (plain HDF5):
```bash
mpirun -n 4 ./build/apps/compression/gys_compress \
  apps/compression/<PARAMS> apps/compression/pdi_out.yaml
```

With in-situ diagnostics (requires a running Dask cluster):
```bash
./apps/compression/deisa-dask_launch_script.sh [SIMU_NODES] [DASK_WORKERS] [GYSELA_PARAMS]
```

The launch script defaults: `SIMU_NODES=1`, `DASK_WORKERS=1`, `GYSELA_PARAMS=params_landau_damping.yaml`, `PDI_CONFIG=pdi_out_diags.yaml`.

## Input constraints

- `Algorithm.nbiter` must be positive.
- `compression_period` must be positive, smaller than `nbiter`, and a multiple of `nbstep_diag = time_diag / deltat`.

## Restart logic

The first segment starts from the analytic initial condition (`nb_restart = 0`). Each subsequent segment reads its approximate restart file via `fdistribu_filename` and continues from `iter_offset`. The launcher writes a fresh YAML config per segment with the appropriate values.

## Compression-event manifest

Each entry in `compression_events.yaml` records: segment id, absolute iteration, file index, restart paths, sizes, PCA explained variance, reconstruction errors, compression ratio, and cleanup flags.

## Evaluating a run

```bash
python apps/compression/evaluate_compression.py diagnostics <data_dir>/branch_baseline/diagnostics.csv <data_dir>/branch_compressed/diagnostics.csv -o compression_analysis.png
```

Reads one or more `diagnostics.csv` files (deduplicating restart-boundary rows) and overlays them; omit `-o` to show interactively instead of saving.

## Online (in-situ) neural-network compressor

`OnlineNeuralNetworkCompressor` (`src/python/compression_methods/neural_network.py`) fits one small INR per species directly on each MPI rank's local `fdistribu` chunk, warm-started across calls. Every call from `apply_online_compression` also writes `params_iterXXXXX_rankXXX.npz` into the run's data directory, containing that rank's current network weights plus the local data chunk it was just fit on.

Evaluate a saved snapshot:

```bash
python apps/compression/evaluate_compression.py online-networks <data_dir> --iter <ITER> --rank <RANK>
```

`--rank` takes one or more ranks, or can be omitted entirely to use every rank found for that iteration. The top row loads every rank's `.npz` for that iteration, tiles their local data into the full global domain, and compares it against the partition-of-unity reconstruction (`assemble_global_field`) assembled from all ranks' networks, with one colored box per requested rank marking where its subdomain sits within the full domain. Each requested rank then gets its own row below, plotting its own network against the local data it was trained on, framed in the same color as its box in the top row -- so `--rank 1 2 3` produces a 4-row figure. Use `--species` and `--plane {xvx,xy,vxvy}` (default `xvx`) to pick which species and 2D slice to plot. The plot is always written into `<data_dir>` as `eval_iter<ITER>_rank<RANK[-RANK...]>_species<SPECIES>_<PLANE>.png`.

## PCA compressor

`fdistribu[sp, x, y, vx, vy]` is reshaped to `(sp*x*y, vx*vy)` and compressed along velocity dimensions. Default settings in `launch_benchmark.py`:

```python
COMPRESSOR_CLASS = PCACompressor
COMPRESSOR_PARAMS = {"n_components": 8, "normalisation": "none", "clip_nonnegative": False}
```

## Adding a custom compression method

1. Build `gys_compress`.
2. Adjust `params_landau_damping.yaml` (or set `SOURCE_GYSELA_YAML`) with `Algorithm.nbiter`, `Algorithm.deltat`, `Output.time_diag`, and `CompressionBenchmark.compression_period`.
3. Implement your compressor in `src/python/compression_methods/` using the `Compressor` blueprint.
4. Set `COMPRESSOR_CLASS` and `COMPRESSOR_PARAMS` in `launch_benchmark.py`.
5. Run `launch_benchmark.py`, then `evaluate_compression.py diagnostics`.
