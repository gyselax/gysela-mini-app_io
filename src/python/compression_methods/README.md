# Compression methods

Implicit Neural Representation (INR) and PCA/POD compressors for
`fdistribu[species, x, y, vx, vy]`. Two entry points into `neural_network.py`:

- `NeuralNetworkCompressor` -- **offline**: fits one INR per species once per checkpoint,
  over the whole assembled global grid (single process, whole GPU).
- `OnlineNeuralNetworkCompressor` -- **online**: fits one INR per species per checkpoint,
  independently on each MPI rank's own local `fdistribu` chunk (4 concurrent ranks sharing
  one physical GPU on persee -- see [Multi-rank GPU contention](#multi-rank-gpu-contention-online-only)).

`PCA.py` implements the POD/PCA baseline (`--compression POD`). All results below are from
the `two_stream` case at the tested grid, `x,y,vx,vy = 64,64,65,65`; they should generalize
to `landau_damping` but haven't been checked there.

## Architectures

All INR architectures are SIREN or Fourier-feature MLPs (`AVAILABLE_INR_ARCHS` in
`neural_network.py`), 4 inputs (`x,y,vx,vy`, normalized), 1 output. Fourier MLP variants
exist in the code but have never been used in this project -- everything below is SIREN.

| `arch` | hidden layers | n_params |
|---|---|---|
| `periodic_siren_small_32` | 3 x 32 | 2,369 |
| `periodic_siren_small_32_l5` | 5 x 32 | 4,481 |
| `periodic_siren_small_32_l8` | 8 x 32 | 7,649 |
| `periodic_siren_small_16_l3` | 3 x 16 | 673 |
| `periodic_siren_small_16_l5` | 5 x 16 | 1,217 |
| `periodic_siren_small_16_l8` | 8 x 16 | 2,033 |
| `periodic_siren_deep_128` | 5 x 128 | 67,073 |
| `periodic_fourier_mlp_small_32` | 8 Fourier features, 3 x 32 | 2,737 (untested) |
| `periodic_fourier_mlp_deep_128` | 16 Fourier features, 5 x 128 | 70,497 (untested) |

## Best architecture / accuracy / compression ratio

At `64,64,65,65` (never measured at `128,128,129,129` -- the INR payload size doesn't depend
on grid resolution, only on `arch`, so the ratio should scale roughly with the grid point
count, ~15.8x going to `128,128,129,129`, but that's a projection, not a measurement):

**Offline** -- every architecture below was run with both Gauss-Newton (`gn_iters=150`) and
L-BFGS (`lbfgs_iters=50`), except `deep_128` which is L-BFGS-only because Gauss-Newton cannot
be used with it at all (explained right after the table). All numbers below -- offline and
online alike -- are measured on the Xeon(physics)/GPU-V100(training) split node (see
[Multi-rank GPU contention](#multi-rank-gpu-contention-online-only))

| arch | polish optimizer | avg rel_l2 | avg time/checkpoint | compression ratio |
|---|---|---|---|---|
| `small_32` | Gauss-Newton | 0.72% | 104s | 7535x |
| `small_32` | L-BFGS | 3.03% | 284s | 7535x |
| **`small_32_l5` (best)** | **Gauss-Newton** | **0.18%** | 176s | 4063x |
| `small_32_l5` | L-BFGS | 1.69% | 348s | 4063x |
| `small_32_l8` | Gauss-Newton | 16.2% (overfits) | 634s | 2412x |
| `small_32_l8` | L-BFGS | 9.75% (overfits less, see note) | 485s | 2412x |
| `small_16_l3` | Gauss-Newton | 3.32% | 48s | 24468x |
| `small_16_l3` | L-BFGS | 7.12% | 233s | 24468x |
| `small_16_l5` | Gauss-Newton | 0.55% | 87s | 14319x |
| `small_16_l5` | L-BFGS | 4.36% | 280s | 14319x |
| `small_16_l8` (alternative) | Gauss-Newton | 0.32% | 88s | 8783x |
| `small_16_l8` | L-BFGS | 4.76% | 343s | 8783x |
| `deep_128` (highest-capacity arch) | L-BFGS (GN forbidden) | 3.78% | 633s | 277x |

Gauss-Newton beats L-BFGS on every architecture except `small_32_l8`, where the pattern
flips (16.2% GN vs 9.75% L-BFGS): `small_32_l8` overfits under GN -- 8 SIREN layers have
enough capacity to fit a mini-batch of points near-exactly without that fit being consistent
across mini-batches, not a memory issue -- and L-BFGS's weaker, history-approximated
curvature happens to act as a mild regularizer against exactly that failure mode (see
[Gauss-Newton vs L-BFGS](#gauss-newton-vs-l-bfgs)). `small_32_l5` is the offline default;
`small_16_l8` trades a little accuracy for a much smaller payload if that matters more.

`deep_128` is architecturally our most expressive/flexible network (67,073 params, 5 layers
of 128 vs. `small_32_l5`'s 4,481 params) -- it just can't be paired with Gauss-Newton: GN's
normal-equations matrix `JᵀJ` is `n_params x n_params` and dense, so it scales with the
*square* of the parameter count, independent of `gn_chunk_size` (chunking only bounds the
Jacobian computation that builds `JᵀJ`, not the matrix itself once assembled). For
`deep_128` that matrix alone needs ~36GB (the gpu: persee/v100), so `neural_network.py` raises immediately if
`polish_optimizer="gauss_newton"` is requested with any `deep_128` arch -- L-BFGS never
forms that matrix, so it has no equivalent wall and is the only usable polish optimizer for
`deep_128`. Despite that capacity edge, `deep_128`+L-BFGS still loses badly to every
`small_*`+GN row above: curvature quality (GN's exact, freshly-computed Jacobian every
iteration vs. L-BFGS's approximate, historical one -- see
[Gauss-Newton vs L-BFGS](#gauss-newton-vs-l-bfgs)) matters more here than raw network
capacity.

**Online** -- 4 concurrent MPI ranks, one local INR per rank, global l2-weighted
`relative_l2_error` across ranks. `small_32`/`small_32_l5`/`small_16_l8` use Gauss-Newton
(`warm_iters_gn=200` cold-start, `refine_iters_gn=150` warm-started, this session's
confirmed default). `deep_128` cannot use Gauss-Newton online either, for the identical
`JᵀJ`-memory reason as offline -- it's shown with chunked L-BFGS instead.

| arch | polish optimizer | avg rel_l2 (final checkpoint) | avg time/checkpoint (cold / warm) |
|---|---|---|---|
| `small_32` | Gauss-Newton | 0.52% | ~338s / ~209s |
| `small_32_l5` | Gauss-Newton | 0.46% | ~981s / ~675s |
| `small_16_l5` | Gauss-Newton | 3.58% | ~317s / ~137-140s |
| `small_16_l8` | Gauss-Newton | 5.78% | ~441s / ~267s |
| `small_32_l8` | Gauss-Newton | **abandoned** -- already fails offline (18.8%, overfitting) and `gn_chunk_size=150` makes it impractically slow online (~35min cold-start); not being retested | -- |
| `small_32_l5` | L-BFGS (chunked) | 9.71% -- structural ceiling, not a budget issue | ~538s / ~64-260s\* |
| `small_16_l8` | L-BFGS (chunked) | 16.8% | ~575s / ~76-86s |
| `small_16_l5` | L-BFGS (chunked) | 19.5% -- worst L-BFGS result online | ~611s / ~61-69s |
| `deep_128` (highest-capacity arch) | L-BFGS (GN forbidden) | 9.90%, not reproducible run-to-run | ~1134s / ~121-172s |

\* L-BFGS's warm-call time online depends on `refine_iters_lbfgs`: ~64s at the retained
value of 10, ~260s if pushed to 50 -- which was tested and gave no accuracy gain, so 10
stays the default (see [Warm/cold iteration budgets](#warmcold-iteration-budgets)).

Online is noisier than offline because each rank only sees its own local velocity-space
quadrant, not the whole domain.

**Recommendation: `small_32_l5` + Gauss-Newton, both modes.**

## Gauss-Newton vs L-BFGS

Not a batch-size question -- it's curvature quality. L-BFGS approximates the inverse Hessian
from a rolling history of ~10-20 past (gradient, position) pairs (a secant approximation,
potentially stale). Gauss-Newton (`train_map_gn`) recomputes the exact Jacobian every
iteration (`jax.jacfwd`) and derives the dense `JᵀJ` from it -- never stale. SIREN's
oscillatory loss landscape favors the exact, fresh curvature: `small_32+GN` beats
`deep_128+LBFGS` (a much bigger network) both on accuracy and speed (see table above), and
online, chunked L-BFGS plateaus at ~10x worse than GN on the same architecture regardless of
how much iteration budget it's given (tested up to 5x the default, no change -- the ceiling
is structural, not a budget problem).

Tradeoff: GN's dense `JᵀJ` (`n_params x n_params`) needs ~36GB for `deep_128` alone,
independent of chunking -- `neural_network.py` raises on
`polish_optimizer="gauss_newton"` for any `deep_128` arch. L-BFGS never forms a dense
matrix, so it has no such ceiling (but chunking is still needed for the batch itself, next
section).

## Chunking (avoiding OOM)

`gn_chunk_size` bounds the peak memory of the GN Jacobian computation, independent of
`gn_n_map` (which only sets the mini-batch size / statistical conditioning, not memory).
`lbfgs_chunk_size` does the same for L-BFGS's full-batch loss. Both process the batch
through `jax.lax.scan` + `jax.checkpoint` in fixed-size pieces, discarding each piece's
activations once used -- the result is bit-identical to the unchunked computation (verified
to ~1e-17 float64 noise), not an approximation.

### Multi-rank GPU contention (online only)

Offline is single-process: one GN fit gets the whole GPU. Online runs 4 concurrent MPI ranks
on the *same physical GPU* (one GPU on persee: V100), each its own JAX process -- a different
contention than the physics-vs-training one already fixed by the CPU/GPU split (physics on
Xeon, training on GPU). This is why online needs much smaller `gn_chunk_size` than offline
for the same architecture.

### Known-safe values

| | offline `gn_chunk_size` | online `gn_chunk_size` | `lbfgs_chunk_size` (both modes) |
|---|---|---|---|
| `small_32`, `small_32_l5`, `small_16_*` | 2000 | 400 | 200000 |
| `small_32_l8` | 1000 | 150 (slow, ~35min cold-start) | 200000 |
| `deep_128` | GN forbidden | GN forbidden | 200000 |

`lbfgs_chunk_size=200000` has never needed lowering, either mode, any architecture tested
including `deep_128` -- L-BFGS's own memory footprint per point is far smaller than GN's
Jacobian, so it tolerates a much larger chunk under the same contention.

**Open**: `deep_128` + chunked L-BFGS online showed accuracy degrading continuously across
checkpoints (~3x by the last one) in a 2026-08-26 run, but a 2026-08-27 rerun with the
identical config instead plateaued (~9.9% from checkpoint 40 onward, no further drift) --
not reproducible, cause unknown. See the "Online sous contention" artifact for the full
history.

## Warm/cold iteration budgets

Both compressors keep the network warm-started across checkpoints (`self.models` persists);
the first call per local grid shape ("cold") gets a larger iteration budget than later ones
("warm"), which only need to track how the physics moved since the last call.

| | offline | online |
|---|---|---|
| ADAM cold / warm | `max_iters` / `warm_max_iters` (GN only -- L-BFGS keeps `max_iters` on warm too) | `warm_iters_adam` / `refine_iters_adam` (always reduced) |
| polish cold / warm | `lbfgs_iters` / `gn_iters` -- fixed, no reduction | `warm_iters_lbfgs` / `warm_iters_gn` vs `refine_iters_lbfgs` / `refine_iters_gn` -- reduced |

Retained defaults: offline `GN(gn_iters=150, warm_max_iters=500)`; online
`warm_iters_gn=200, refine_iters_gn=150` (raising `refine_iters_lbfgs` past 10 was tested and
gave no accuracy gain, just more compute time).

## Run directory layout

```text
results_<case_name>/                        # e.g. results_two_stream/
├── branch_baseline/                         # shared, uncompressed reference sim
├── offline_compression/
│   └── NN/<arch>/polish_<optimizer>/        # or POD/r<n_components>/
│       ├── compression_events_offline.csv
│       ├── loss_histories/loss_iter<ITER>_sp<SP>_<arch>.npy
│       └── payload_iter<ITER>.npz
└── online_compression/                      # same shape, minus POD
    └── NN/<arch>/polish_<optimizer>/
        ├── compression_events_rank<NNN>.csv     # one per MPI rank
        ├── loss_histories/loss_iter<ITER>_sp<SP>_rank<NNN>_<arch>.npy
        └── params_iter<ITER>_rank<NNN>.npz
```

Full layout, comparison figures (`--frob`, `--checkpoint-time`, `--inr-loss`, ...) and the
`evaluate_compression.py compare` CLI are documented in
[`apps/compression/README.md`](../../../apps/compression/README.md).

## Example commands

```bash
# offline, INR (Gauss-Newton)
python apps/compression/launch_benchmark.py results_two_stream --overwrite \
  --compression NN --arch-nn periodic_siren_small_32_l5 \
  --polish-optimizer-nn gauss_newton --dask-workers 1

# offline, POD
python apps/compression/launch_benchmark.py results_two_stream --overwrite \
  --compression POD --rank 8 --dask-workers 1

# online, INR (Gauss-Newton) -- 4 MPI ranks, one local fit each
python apps/compression/launch_benchmark.py results_two_stream --overwrite \
  --online --compression NN --arch-nn periodic_siren_small_32_l5 \
  --polish-optimizer-nn gauss_newton --dask-workers 1
```
