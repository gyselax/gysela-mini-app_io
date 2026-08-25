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
results_<case_name>/                        # e.g. results_two_stream/
├── branch_baseline/                        # shared, uncompressed reference sim -- never duplicated
├── config_baseline.yaml
├── compression_events.yaml                 # legacy manifest, last-branch-wins, minor quirk
├── pdi_out_diags.yaml
├── offline_compression/
│   ├── NN/<arch>/polish_<optimizer>/       # e.g. NN/periodic_siren_small_32_l5/polish_gauss_newton/
│   ├── POD/r<n_components>/                # offline only -- online has no POD path
│   ├── comparisons/{baseline_vs_INR,baseline_vs_POD,mixed_comparisons}/
│   ├── config_NN_<arch>_polish_<optimizer>.yaml, config_POD_r<n>.yaml   # per-case config snapshots
│   └── diags_comparison.png
└── online_compression/                     # same shape as offline_compression, minus POD;
                                             # created automatically on first --online run
```

`branch_baseline/`, `config_baseline.yaml`, `compression_events.yaml` and `pdi_out_diags.yaml` are the only things shared between the two pipelines -- everything else (results, comparisons, config snapshots, the aggregate `diags_comparison.png`) is fully separated under `offline_compression/` vs `online_compression/`. The baseline suppresses all intermediate HDF5 snapshots; conserved variables come entirely from in-situ diagnostics.

Inside one `NN/<arch>/polish_<optimizer>/` (or `POD/r<n>/`) case directory, offline and online differ because offline fits one compressor over the whole assembled grid while online fits one compressor per MPI rank on its own local chunk (see the compression_methods README for that distinction):

| | offline | online |
| --- | --- | --- |
| per-checkpoint metrics | `compression_events_offline.csv` | `compression_events_rank<NNN>.csv`, one per MPI rank |
| training loss curves | `loss_histories/loss_iter<ITER>_sp<SP>_<arch>.npy` | `loss_histories/loss_iter<ITER>_sp<SP>_rank<NNN>_<arch>.npy`, one per rank |
| saved network weights | `payload_iter<ITER>.npz` | `params_iter<ITER>_rank<NNN>.npz`, one per rank |
| snapshots | `GYSELALIBXX_compressed_<ITER>.h5` per checkpoint + final `GYSELALIBXX_<ITER>.h5` | final `GYSELALIBXX_<ITER>.h5` only |
| debug plots | `final_snapshot_comparison.png` | `final_snapshot_comparison.png` + `nn_online_local_rank<NNN>_call<CALL>_xvx.png` per rank per call |

`compression_events_offline.csv` / `compression_events_rank*.csv` are the primary place to check compression quality (`relative_l2_error`, `compression_seconds`/`decompression_seconds`, etc. -- see the `--frob`/`--checkpoint-time` figures above). **Caveat**: `final_loss_per_species` in that CSV is `jnp.min()` over the *entire* loss history regardless of which model was actually deployed -- cross-check against `relative_l2_error` (the honest end-to-end metric), don't trust `final_loss` alone.

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

## Cross-run comparisons and figures (`compare`)

```bash
python apps/compression/evaluate_compression.py compare \
  --params <run_dir>/case1/diagnostics.csv <run_dir>/case2/diagnostics.csv ... \
  --frob --frob-pod --frob-inr --checkpoint-time --svd-spectrum --inr-loss \
  [-o <output_dir>]
```

Each `--params` entry is the path to a case's `diagnostics.csv` (not just the directory) -- its parent directory is where the compressor's own outputs (`compression_events*.csv`, `loss_histories/`, `svd_spectrums/`, ...) are expected to live. Every flag below is independent; pass only the ones you want. If a case is missing the data a flag needs (e.g. `--svd-spectrum` on an NN case, which has no `svd_spectrums/`), that flag is silently skipped for that case, not for the whole run.

Without `-o`, the output directory is resolved automatically: for a single `--params` path, `<run_dir>/comparisons/`; for several paths sharing the same `offline_compression/`or `online_compression/` root, `<that_root>/comparisons/<baseline_vs_INR|baseline_vs_POD|mixed_comparisons>/` depending on whether the cases are NN, POD, or a mix.

| Flag | Figure | What it shows | Written to |
| --- | --- | --- | --- |
| `--frob` | `frob_error_comparison.png` | Relative L2 (Frobenius) reconstruction error vs. checkpoint iteration, one line per case, log scale. The core accuracy-over-time comparison across architectures/optimizers/ranks (`compression_events*.csv`'s `relative_l2_error`, averaged across MPI ranks for online cases). | `<out_dir>/global/` |
| `--frob-pod` | `frob_error_pod.png` | Same plot, filtered to cases whose label contains `pod`. | `<out_dir>/global/` |
| `--frob-inr` | `frob_error_nn.png` | Same plot, filtered to cases whose label contains `nn` (i.e. every INR/NN case). | `<out_dir>/global/` |
| `--checkpoint-time` | `checkpoint_time_<case_label>.png` | One figure **per case** (not overlaid): a stacked bar per checkpoint, compression time below decompression time, each bar annotated in seconds and minutes, total run time annotated on the figure. The wall-clock-cost counterpart to `--frob`'s accuracy view. | `<out_dir>/<online\|offline>/NN/<arch>/<optimizer>/` or `.../POD/r<n>/` (mirrors the case's own subtree) |
| `--svd-spectrum` | `svd_spectrum.png` | Singular-value decay ($\sigma_i/\sigma_1$, log scale) of the **last checkpoint** for every POD case that has a `svd_spectrums/` directory, one line per case with a dashed vertical line at the retained rank (`param_n_components`). Only meaningful for POD -- NN cases produce nothing here. | `<out_dir>/global/` |
| `--inr-loss` | `loss_iter<ITER>_sp<SP>[_rank<NNN>]_<arch>.png` (one per file) | ADAM-then-polish training loss curve (MSE, log scale) for one species at one checkpoint (one rank, for online), read from that case's `loss_histories/*.npy`. The x-axis label names the polish optimizer used (`Gauss-Newton` or `L-BFGS`, inferred from the path). Only meaningful for NN -- POD cases have no `loss_histories/`. | `<out_dir>/<online\|offline>/NN/<arch>/<optimizer>/` (one file per file already present in that case's `loss_histories/`) |

### Concrete example

Comparing two offline Gauss-Newton architectures against each other (from this repo's own `results_two_stream/`):

```bash
python apps/compression/evaluate_compression.py compare \
  --params \
    results_two_stream/offline_compression/NN/periodic_siren_small_32/polish_gauss_newton/diagnostics.csv \
    results_two_stream/offline_compression/NN/periodic_siren_small_32_l5/polish_gauss_newton/diagnostics.csv \
  --frob --checkpoint-time --inr-loss
```

writes `results_two_stream/offline_compression/comparisons/baseline_vs_INR/global/frob_error_comparison.png` (both archs overlaid) plus per-case `checkpoint_time_*.png` and `loss_iter*.png` under each arch's own `NN/<arch>/polish_gauss_newton/` subtree. Add a POD case's `diagnostics.csv` to the same `--params` list and `--svd-spectrum` becomes meaningful too (and the output root becomes `mixed_comparisons/` since the cases now mix NN and POD).

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
