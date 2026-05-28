import json
import os
import shutil

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCACompressor:
    """
    PCA compressor for GYSELALIBXX fdistribu restart fields.

    The expected input dataset layout is:

        fdistribu[species, x, y, vx, vy]

    The PCA matrix representation is:

        rows    = species * x * y
        columns = vx * vy
    """

    ALLOWED_NORMALISATIONS = {"none", "zscore", "log", "asinh"}

    def __init__(
        self,
        n_components=32,
        normalisation="none",
        alpha=1e-6,
        clip_nonnegative=False,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.normalisation = normalisation.lower()
        self.alpha = float(alpha)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.random_state = random_state

        if self.normalisation not in self.ALLOWED_NORMALISATIONS:
            raise ValueError(
                f"Unknown normalisation '{self.normalisation}'. "
                f"Expected one of {sorted(self.ALLOWED_NORMALISATIONS)}."
            )

        self.scaler = StandardScaler() if self.normalisation == "zscore" else None
        self.model = None
        self.original_shape = None

    # -------------------------------------------------------------------------
    # Shape conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def array_to_matrix(f):
        """
        Convert fdistribu[species, x, y, vx, vy] into a 2D matrix.

        Returns
        -------
        X:
            Matrix of shape (Nspecies * Nx * Ny, Nvx * Nvy).
        original_shape:
            Original 5D shape needed for reconstruction.
        """
        f = np.asarray(f, dtype=np.float64)

        if f.ndim != 5:
            raise ValueError("Expected fdistribu with rank 5 " f"(Nspecies, Nx, Ny, Nvx, Nvy), got shape {f.shape}.")

        original_shape = f.shape
        n_species, nx, ny, nvx, nvy = original_shape

        return f.reshape(n_species * nx * ny, nvx * nvy), original_shape

    @staticmethod
    def matrix_to_array(X, original_shape):
        return np.asarray(X).reshape(tuple(original_shape))

    # -------------------------------------------------------------------------
    # Normalisation
    # -------------------------------------------------------------------------

    def _preprocess(self, X, fit):
        if self.normalisation == "none":
            return X

        if self.normalisation == "log":
            return np.log10(np.clip(X, 1e-16, None))

        if self.normalisation == "asinh":
            return np.arcsinh(X / self.alpha)

        if self.normalisation == "zscore":
            if self.scaler is None:
                raise RuntimeError("Z-score normalisation requested without scaler.")

            if fit:
                return self.scaler.fit_transform(X)

            return self.scaler.transform(X)

        raise RuntimeError(f"Unhandled normalisation: {self.normalisation}")

    def _inverse_preprocess(self, X):
        return self.inverse_preprocess(
            X,
            normalisation=self.normalisation,
            alpha=self.alpha,
            scaler_mean=None if self.scaler is None else self.scaler.mean_,
            scaler_scale=None if self.scaler is None else self.scaler.scale_,
        )

    @staticmethod
    def inverse_preprocess(
        X,
        normalisation,
        alpha=1e-6,
        scaler_mean=None,
        scaler_scale=None,
    ):
        normalisation = normalisation.lower()

        if normalisation == "none":
            return X

        if normalisation == "log":
            return 10.0**X

        if normalisation == "asinh":
            return alpha * np.sinh(X)

        if normalisation == "zscore":
            if scaler_mean is None or scaler_scale is None:
                raise RuntimeError("Z-score inverse preprocessing requires scaler_mean and scaler_scale.")

            return X * scaler_scale + scaler_mean

        raise RuntimeError(f"Unhandled normalisation: {normalisation}")

    # -------------------------------------------------------------------------
    # In-memory compression / decompression
    # -------------------------------------------------------------------------

    def compress_array(self, f):
        X, original_shape = self.array_to_matrix(f)

        max_components = min(X.shape)

        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large for matrix "
                f"shape {X.shape}. Maximum allowed value is {max_components}."
            )

        self.original_shape = original_shape

        X_proc = self._preprocess(X, fit=True)

        self.model = PCA(
            n_components=self.n_components,
            svd_solver="auto",
            random_state=self.random_state,
        )

        return self.model.fit_transform(X_proc)

    def decompress_array(self, coefficients):
        if self.model is None:
            raise RuntimeError("No fitted PCA model available. Call compress_array first.")

        if self.original_shape is None:
            raise RuntimeError("original_shape is not available.")

        X_approx_proc = self.model.inverse_transform(coefficients)
        X_approx = self._inverse_preprocess(X_approx_proc)

        if self.clip_nonnegative:
            X_approx = np.clip(X_approx, 0.0, None)

        return self.matrix_to_array(X_approx, self.original_shape)

    # -------------------------------------------------------------------------
    # NPZ payload I/O
    # -------------------------------------------------------------------------

    def save_compressed_payload(self, compressed_path, coefficients):
        if self.model is None:
            raise RuntimeError("No fitted PCA model available.")

        if self.original_shape is None:
            raise RuntimeError("original_shape is not available.")

        os.makedirs(os.path.dirname(os.path.abspath(compressed_path)), exist_ok=True)

        metadata = {
            "normalisation": self.normalisation,
            "alpha": self.alpha,
            "clip_nonnegative": self.clip_nonnegative,
            "random_state": self.random_state,
            "layout": "fdistribu[species,x,y,vx,vy]",
            "matrix_layout": "rows=species*x*y, columns=vx*vy",
        }

        payload = {
            "coefficients": coefficients,
            "components": self.model.components_,
            "mean": self.model.mean_,
            "explained_variance": self.model.explained_variance_,
            "explained_variance_ratio": self.model.explained_variance_ratio_,
            "singular_values": self.model.singular_values_,
            "original_shape": np.array(self.original_shape, dtype=np.int64),
            "n_components": np.array([self.n_components], dtype=np.int64),
            "metadata_json": np.array(json.dumps(metadata)),
        }

        if self.normalisation == "zscore":
            payload["scaler_mean"] = self.scaler.mean_
            payload["scaler_scale"] = self.scaler.scale_
            payload["scaler_var"] = self.scaler.var_

        np.savez_compressed(compressed_path, **payload)

    @staticmethod
    def load_compressed_payload(compressed_path):
        if not os.path.exists(compressed_path):
            raise FileNotFoundError(compressed_path)

        with np.load(compressed_path, allow_pickle=False) as payload:
            data = {key: payload[key] for key in payload.files}

        metadata_json = data.get("metadata_json")

        if metadata_json is None:
            metadata = {}
        else:
            metadata = json.loads(str(metadata_json.item()))

        data["metadata"] = metadata

        return data

    @classmethod
    def reconstruct_array_from_payload(cls, compressed_path):
        payload = cls.load_compressed_payload(compressed_path)

        coefficients = payload["coefficients"]
        components = payload["components"]
        mean = payload["mean"]
        original_shape = tuple(payload["original_shape"].astype(int))

        metadata = payload["metadata"]

        normalisation = metadata.get("normalisation", "none")
        alpha = float(metadata.get("alpha", 1e-6))
        clip_nonnegative = bool(metadata.get("clip_nonnegative", False))

        X_approx_proc = coefficients @ components + mean

        scaler_mean = payload.get("scaler_mean")
        scaler_scale = payload.get("scaler_scale")

        X_approx = cls.inverse_preprocess(
            X_approx_proc,
            normalisation=normalisation,
            alpha=alpha,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

        if clip_nonnegative:
            X_approx = np.clip(X_approx, 0.0, None)

        return cls.matrix_to_array(X_approx, original_shape)

    # -------------------------------------------------------------------------
    # HDF5 restart-file workflow
    # -------------------------------------------------------------------------

    def compress_decompress_h5(
        self,
        input_h5,
        output_h5,
        compressed_path=None,
        dataset_name="fdistribu",
    ):
        """
        Compress and immediately reconstruct the fdistribu dataset.

        The full HDF5 file is copied first, then only `dataset_name` is replaced
        by its PCA reconstruction. This preserves the restart-file container,
        datasets, metadata, and PDI-compatible structure.
        """
        if not os.path.exists(input_h5):
            raise FileNotFoundError(input_h5)

        os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)

        shutil.copy2(input_h5, output_h5)

        with h5py.File(output_h5, "r+") as h5:
            if dataset_name not in h5:
                raise KeyError(f"Dataset '{dataset_name}' not found in {output_h5}.")

            f_original = h5[dataset_name][:]
            coefficients = self.compress_array(f_original)
            f_reconstructed = self.decompress_array(coefficients)

            if h5[dataset_name].shape != f_reconstructed.shape:
                raise RuntimeError(
                    "Reconstructed shape mismatch: "
                    f"expected {h5[dataset_name].shape}, "
                    f"got {f_reconstructed.shape}."
                )

            h5[dataset_name][...] = f_reconstructed

        if compressed_path is not None:
            self.save_compressed_payload(compressed_path, coefficients)

        return self.compute_metrics(
            f_original=f_original,
            f_reconstructed=f_reconstructed,
            input_h5=input_h5,
            output_h5=output_h5,
            compressed_path=compressed_path,
        )

    @classmethod
    def reconstruct_h5_from_payload(
        cls,
        template_h5,
        output_h5,
        compressed_path,
        dataset_name="fdistribu",
    ):
        """
        Reconstruct a restart HDF5 file from:

            template_h5 + compressed npz -> output_h5

        The NPZ reconstructs only fdistribu. The template H5 provides the HDF5
        container structure expected by the solver.
        """
        if not os.path.exists(template_h5):
            raise FileNotFoundError(template_h5)

        f_reconstructed = cls.reconstruct_array_from_payload(compressed_path)

        os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
        shutil.copy2(template_h5, output_h5)

        with h5py.File(output_h5, "r+") as h5:
            if dataset_name not in h5:
                raise KeyError(f"Dataset '{dataset_name}' not found in {output_h5}.")

            if h5[dataset_name].shape != f_reconstructed.shape:
                raise RuntimeError(
                    "Reconstructed shape mismatch: "
                    f"expected {h5[dataset_name].shape}, "
                    f"got {f_reconstructed.shape}."
                )

            h5[dataset_name][...] = f_reconstructed

        return output_h5

    def compute_metrics(
        self,
        f_original,
        f_reconstructed,
        input_h5=None,
        output_h5=None,
        compressed_path=None,
    ):
        diff = f_original - f_reconstructed

        original_norm = np.linalg.norm(f_original.ravel())

        relative_l2_error = np.linalg.norm(diff.ravel()) / original_norm if original_norm > 0.0 else np.nan

        max_abs_error = np.max(np.abs(diff))

        original_size = os.path.getsize(input_h5) if input_h5 is not None else None
        reconstructed_size = os.path.getsize(output_h5) if output_h5 is not None else None
        compressed_size = (
            os.path.getsize(compressed_path)
            if compressed_path is not None and os.path.exists(compressed_path)
            else None
        )

        compression_ratio = (
            original_size / compressed_size
            if original_size is not None and compressed_size is not None and compressed_size > 0
            else None
        )

        return {
            "input_h5": input_h5,
            "output_h5": output_h5,
            "compressed_path": compressed_path,
            "n_components": self.n_components,
            "normalisation": self.normalisation,
            "explained_variance_ratio_sum": (
                float(np.sum(self.model.explained_variance_ratio_)) if self.model is not None else None
            ),
            "relative_l2_error": float(relative_l2_error),
            "max_abs_error": float(max_abs_error),
            "original_size": original_size,
            "reconstructed_size": reconstructed_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
        }
