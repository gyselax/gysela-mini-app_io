import dask.array as da
import numpy as np
from dask_ml.decomposition import IncrementalPCA
from distributed import wait

from compression_methods.PCA import PCACompressor


class IncrementalPCACompressor(PCACompressor):
    """
    dask-ml IncrementalPCA compressor for GYSELALIBXX fdistribu fields.

    Same matrix representation as PCACompressor:

        rows    = species * x * y
        columns = vx * vy

    but the model is fitted one row batch at a time, and the array stays a
    dask array (the one deisa assembles, chunked along vx and vy) from the fit
    down to the reconstruction, so the global field is never resident in the
    analytics process. The vx, vy chunks are gathered into whole columns by the
    dask workers, batch_size rows at a time.
    """

    method_name = "IncrementalPCA"
    accepts_dask = True

    def __init__(
        self,
        n_components=32,
        normalisation="none",
        alpha=1e-6,
        clip_nonnegative=False,
        batch_size=None,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            normalisation=normalisation,
            alpha=alpha,
            clip_nonnegative=clip_nonnegative,
            random_state=random_state,
        )

        self.batch_size = None if batch_size is None else int(batch_size)
        self.method_name = type(self).method_name
        self.params["batch_size"] = self.batch_size
        self.row_chunks = None

    # -------------------------------------------------------------------------
    # Shape conversion (dask, chunked along rows only)
    # -------------------------------------------------------------------------

    def _rows_per_batch(self, nx, ny):
        """x indices per row batch, so that one batch holds about batch_size rows."""
        if self.batch_size is None:
            return nx

        return max(1, min(nx, self.batch_size // ny))

    def array_to_matrix_dask(self, f):
        """
        Dask counterpart of array_to_matrix.

        vx, vy are gathered into a single chunk (a batch needs whole columns)
        and the (species, x) blocks are concatenated in C order, so the rows
        are laid out exactly as in the numpy reshape.
        """
        f = da.asarray(f, dtype=np.float64)

        if f.ndim != 5:
            raise ValueError("Expected fdistribu with rank 5 " f"(Nspecies, Nx, Ny, Nvx, Nvy), got shape {f.shape}.")

        original_shape = f.shape
        _n_species, nx, ny, nvx, nvy = original_shape

        f = f.rechunk((1, self._rows_per_batch(nx, ny), -1, -1, -1))
        blocks = [block.reshape(-1, nvx * nvy) for block in f.blocks.ravel()]

        return da.concatenate(blocks, axis=0), original_shape

    def matrix_to_array_dask(self, X, original_shape, row_chunks):
        """Dask counterpart of matrix_to_array, inverse of array_to_matrix_dask."""
        n_species, _nx, ny, nvx, nvy = original_shape

        X = X.rechunk({0: row_chunks, 1: -1})
        blocks = [block.reshape(1, -1, ny, nvx, nvy) for block in X.blocks.ravel()]
        per_species = len(blocks) // n_species

        return da.concatenate(
            [da.concatenate(blocks[i * per_species : (i + 1) * per_species], axis=1) for i in range(n_species)],
            axis=0,
        )

    @staticmethod
    def persist_on_workers(array):
        """Materialise a dask array on the workers and block until it is done, so
        that the timings measured by the base class are the actual work.
        """
        array = array.persist()

        try:
            wait(array)
        except ValueError:  # no distributed client: persist already computed it
            pass

        return array

    # -------------------------------------------------------------------------
    # Normalisation
    # -------------------------------------------------------------------------

    def _preprocess(self, X, fit):
        """Z-score statistics are reduced by dask; the other normalisations are
        elementwise and are applied by the base class.
        """
        if self.normalisation != "zscore":
            return super()._preprocess(X, fit)

        if fit:
            mean, scale = da.compute(X.mean(axis=0), X.std(axis=0))
            self.scaler.mean_ = mean
            self.scaler.var_ = scale**2
            self.scaler.scale_ = np.where(scale > 0.0, scale, 1.0)

        return (X - self.scaler.mean_) / self.scaler.scale_

    # -------------------------------------------------------------------------
    # In-memory compression / decompression
    # -------------------------------------------------------------------------

    def compress_array(self, f):
        X, original_shape = self.array_to_matrix_dask(f)

        max_components = min(X.shape)

        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large for matrix "
                f"shape {X.shape}. Maximum allowed value is {max_components}."
            )

        self.original_shape = original_shape
        self.row_chunks = X.chunks[0]

        X_proc = self.persist_on_workers(self._preprocess(X, fit=True))

        self.model = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        return self.persist_on_workers(self.model.fit_transform(X_proc))

    def decompress_array(self, coefficients):
        if self.model is None:
            raise RuntimeError("No fitted IncrementalPCA model available. Call compress_array first.")

        if self.original_shape is None:
            raise RuntimeError("original_shape is not available.")

        X_approx_proc = self.model.inverse_transform(da.asarray(coefficients))
        X_approx = self._inverse_preprocess(X_approx_proc)

        if self.clip_nonnegative:
            X_approx = da.clip(X_approx, 0.0, None)

        return self.persist_on_workers(self.matrix_to_array_dask(X_approx, self.original_shape, self.row_chunks))

    def save_compressed_payload(self, compressed_path, coefficients):
        super().save_compressed_payload(compressed_path, np.asarray(coefficients))

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _error_metrics(f_original, f_reconstructed):
        """Reduced by the workers, in a single pass over the two dask arrays."""
        f_original = da.asarray(f_original)
        diff = f_original - da.asarray(f_reconstructed)

        original_norm, diff_norm, max_abs_error, mean_abs_error = da.compute(
            da.sqrt(da.sum(f_original**2)),
            da.sqrt(da.sum(diff**2)),
            da.max(da.fabs(diff)),
            da.mean(da.fabs(diff)),
        )

        return {
            "relative_l2_error": float(diff_norm / original_norm) if original_norm > 0.0 else np.nan,
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "rmse": float(diff_norm / np.sqrt(f_original.size)),
        }

    def get_extra_metrics(self):
        return {
            **super().get_extra_metrics(),
            "n_row_batches": None if self.row_chunks is None else len(self.row_chunks),
            "n_samples_seen": getattr(self.model, "n_samples_seen_", None),
        }
