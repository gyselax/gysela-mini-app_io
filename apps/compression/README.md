# GYSELA Compression Mini App

This directory contains a benchmark pipeline for evaluating restart-file compression in a 2D2V Vlasov--Poisson mini-application. The default benchmark uses Landau damping; a two-stream instability case is also available. The workflow compares a reference, uninterrupted simulation against a segmented simulation that periodically compresses and decompresses the distribution function before restarting.

## Main files

| File | Role |
| --- | --- |
| `gys_compress.cpp` | C++ mini-app. Selects the initial condition via `Input.case` (`landau_damping` or `two_stream`), supports cold start and restart, exposes fields and metadata through PDI, and writes `GYSELALIBXX_*.h5` diagnostic/restart files. |
| `launch_benchmark.py` | Orchestrates the Landau-damping compression benchmark using `params_landau_damping.yaml` as the base template. It creates a run directory, runs the baseline branch, runs the periodically restarted compressed branch, performs PCA compression/decompression at restart points, and writes `compression_events.yaml`. |
| `PCA.py` | Implements the PCA compressor for the `fdistribu[species, x, y, vx, vy]` HDF5 dataset. It stores compressed PCA payloads as `.npz` files and reconstructs PDI-compatible HDF5 restart files. |
| `evaluate_compression.py` | Post-processes a completed run. It computes mass, momentum, kinetic energy, potential energy and relative errors with respect to the baseline, then generates `compression_analysis.png`. |
| `params_landau_damping.yaml` | GYSELA/Gyselalib++ input for Landau damping (`Input.case: landau_damping`). Used by `launch_benchmark.py`. Requires `Algorithm.nbiter`, `Algorithm.deltat`, `Output.time_diag`, and `CompressionBenchmark.compression_period`. |
| `params_two_stream.yaml` | GYSELA/Gyselalib++ input for the two-stream case (`Input.case: two_stream`). Set `SpeciesInfo[].mean_velocity_eq` to the stream speed `v0` and `perturb_amplitude` to `eps` in `1 + eps cos(kx x) cos(ky y)`. |
| `pdi_out.yaml` | PDI I/O configuration used by the mini-app to read and write HDF5 data. |

## Pipeline overview

The benchmark creates two branches inside one `compression_run_*` directory.

```text
compression_run_YYYYMMDD_HHMMSS/
├── branch_baseline/
│   └── GYSELALIBXX_*.h5
├── branch_compressed/
│   └── GYSELALIBXX_*.h5
├── periodic_restarts/
│   ├── restart_iter_XXXXX_approx.h5        # optional, normally temporary
│   └── restart_iter_XXXXX_compressed.npz   # optional, normally temporary
├── config_baseline.yaml
└── compression_events.yaml
```

The baseline branch is run once from the analytic initial condition for the full number of iterations.

The compressed branch is run segment by segment. At the end of each segment, except the final one, the current restart file is compressed with your favourite method and immediately reconstructed into an approximate HDF5 restart. The next segment restarts from that approximation.

## Python environment

Activate the project virtual environment (created by `./installer.sh` at the repo root):

```bash
source .gys_env/bin/activate
```

## Running the benchmark

Simply run:
```bash
python apps/compression/launch_benchmark.py
```

By default, this creates a timestamped directory in the project root:

```text
compression_run_YYYYMMDD_HHMMSS
```

You can also provide an explicit output directory:

```bash
python apps/compression/launch_benchmark.py compression_run_pca4
```

The launcher executes the compiled application through MPI:

```bash
mpirun -n 4 ./build/apps/compression/gys_compress <config.yaml> <pdi_out.yaml>
```

Adjust `EXEC_CMD` in `launch_benchmark.py` if the number of ranks or executable path must be changed.

## Running the mini-app directly

From the repository root (after building):

```bash
mpirun -n 4 ./build/apps/compression/gys_compress \
  apps/compression/<PARAMS> \
  apps/compression/pdi_out.yaml
```

where `<PARAMS>` is either `params_landau_damping.yaml` or `params_two_stream.yaml`   
## Launcher options

```bash
python apps/compression/launch_benchmark.py [run_dir] [options]
```

| Option | Description |
| --- | --- |
| `--overwrite` | Allow writing into an existing non-empty run directory. Without this flag, the launcher refuses to reuse a non-empty directory. |
| `--keep-payloads` | Keep `restart_iter_XXXXX_compressed.npz` files. By default, they are removed after their sizes and metrics are stored in `compression_events.yaml`. |
| `--keep-restart-approximations` | Keep `restart_iter_XXXXX_approx.h5` files. By default, each approximation is removed after the next segment consumes it. |
| `--keep-segment-configs` | Keep temporary `config_compressed_segment_XXX.yaml` files. |
| `--keep-pdi-copy` | Keep the copy of `pdi_out.yaml` inside the run directory. |

## Input constraints checked by the launcher

The launcher validates the benchmark configuration before running:

- `Algorithm.nbiter` must be positive.
- `CompressionBenchmark.compression_period` must be positive and smaller than `Algorithm.nbiter`.
- `Output.time_diag / Algorithm.deltat` must be an integer diagnostic step.

These constraints ensure that every compression event occurs at an available diagnostic/restart output.

## Restart logic

The C++ mini-app reads three restart-related entries from the YAML input:

```yaml
Input:
  nb_restart: 0
  fdistribu_filename: none
  iter_offset: 0
```

For the first segment, `nb_restart = 0`, so the simulation starts from the analytic initial condition defined by `Input.case`. For later segments, the launcher writes a temporary YAML file with:

- `nb_restart > 0`
- `fdistribu_filename` set to the reconstructed approximate restart file
- `iter_offset` set to the absolute iteration at which the segment starts

The mini-app then reads `fdistribu` through PDI, transposes it, and continues the simulation.

## Compression-event manifest

At each compression point, `launch_benchmark.py` appends one entry to `compression_events.yaml`. Each entry records:

- segment id
- absolute iteration
- diagnostic file index
- branch restart path
- approximate restart path
- compressed payload path, when kept
- number of PCA components
- raw restart size
- approximate restart size
- compressed payload size
- explained variance ratio sum
- relative L2 reconstruction error
- maximum absolute reconstruction error
- compression ratio
- cleanup flags indicating whether restart artefacts were kept

This manifest is the source for compression statistics when temporary payloads are deleted.

## Evaluating a run

After a benchmark run, execute:

```bash
python apps/compression/evaluate_compression.py compression_run_YYYYMMDD_HHMMSS
```

If no run directory is passed, the evaluator uses the latest `compression_run_*` directory in the current working directory.

To control parallel post-processing:

```bash
python apps/compression/evaluate_compression.py compression_run_pca4 --workers 4
```

The evaluator loads the mesh and time metadata from the baseline branch, processes the HDF5 outputs from both branches, and writes:

```text
compression_analysis.png
```

## Diagnostics computed by the evaluator

For one kinetic species, the evaluator computes:

- mass
- momentum components and momentum norm
- kinetic energy
- electrostatic potential energy, when `electrostatic_potential` is present
- total energy
- relative errors of the compressed branch with respect to the baseline

The analysis figure contains:

1. total energy
2. potential energy
3. total energy variation
4. normalised momentum variation
5. relative mass variation
6. relative errors against the baseline

Compression events are shown as triangular markers along the time axis.

## PCA representation

The compressor expects the restart distribution dataset to have the layout

```text
fdistribu[species, x, y, vx, vy]
```

For PCA, the five-dimensional array is reshaped into a two-dimensional matrix:

```text
rows    = species * x * y
columns = vx * vy
```

Each spatial/species point is therefore treated as one sample, and the local velocity-space distribution is compressed along the velocity dimensions.

The default launcher uses:

```python
from compression_methods.PCA import PCACompressor

COMPRESSOR_CLASS = PCACompressor
COMPRESSOR_PARAMS = {
    "n_components": 8,
    "normalisation": "none",
    "clip_nonnegative": False,
}
```


## How to test your own compression method ?

1. Build `./build/apps/compression/gys_compress`.
2. Check `params_landau_damping.yaml` (or change `SOURCE_GYSELA_YAML` in `launch_benchmark.py` to use another template), especially `Algorithm.nbiter`, `Algorithm.deltat`, `Output.time_diag`, and `CompressionBenchmark.compression_period`.
3. Implement your compressor in the `python/compression_methods` folder using the `Compressor` blueprint.
4. Add your import and set `COMPRESSOR_PARAMS` in `launch_benchmark.py` in the `Compression params / names` section.
5. Run `launch_benchmark.py`.
6. Run `evaluate_compression.py` on the generated run directory.
7. Inspect `compression_events.yaml` and `compression_analysis.png`.

