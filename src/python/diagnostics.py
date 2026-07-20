"""In-situ diagnostics: conserved-variable calculations and deisa analytics callback."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
import numpy as np
from deisa.dask import Deisa
from distributed import get_client

_MEASURE_CFG = None


@dataclass
class PathsConfig:
    data_dir: Path


@dataclass
class GridConfig:
    """Uniform 2D2V mesh (x, y, vx, vy) with precomputed wavenumbers."""

    x:  object
    y:  object
    vx: object
    vy: object
    kx: object = field(init=False)
    ky: object = field(init=False)

    def __post_init__(self):
        Nx = np.asarray(self.x).size
        Ny = np.asarray(self.y).size
        self.kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=self.dy)

    @staticmethod
    def spacing(coord):
        values = np.asarray(coord)
        if values.size < 2:
            raise ValueError(
                "Young Padawan, a single point doth not a grid make. "
                "The Force requires at least two points."
            )
        return float(values[1] - values[0])

    @property
    def dx(self):
        return self.spacing(self.x)

    @property
    def dy(self):
        return self.spacing(self.y)

    @property
    def dvx(self):
        return self.spacing(self.vx)

    @property
    def dvy(self):
        return self.spacing(self.vy)

    @property
    def dV_2D(self):
        return self.dx * self.dy

    @property
    def dV_4D(self):
        return self.dx * self.dy * self.dvx * self.dvy


@dataclass
class Config:
    paths: PathsConfig
    grid:  GridConfig


def to_dask(array):
    return array if isinstance(array, da.Array) else da.from_array(array, name=False)


def init_measure_config(x, y, vx, vy, data_dir="."):
    global _MEASURE_CFG
    _MEASURE_CFG = Config(PathsConfig(Path(data_dir)), GridConfig(x, y, vx, vy))
    return _MEASURE_CFG


def get_measure_config():
    if _MEASURE_CFG is None:
        raise RuntimeError(
            "Padawan, you have not yet trained your config. "
            "Call init_measure_config first, you must."
        )
    return _MEASURE_CFG


def density(fdistribu, grid):
    """Integrate f over all species and velocity dimensions -> n(x, y)."""
    f_da = to_dask(fdistribu)
    return da.sum(f_da, axis=(0, 3, 4)) * grid.dvx * grid.dvy


def poisson_fft(n, grid):
    """Solve Laplacian(phi) = -(n - 1) on a doubly-periodic domain using da.fft.fft2.

    In Fourier space: -(kx^2 + ky^2) * phi_hat(k) = rho_hat(k)
    phi_hat(0,0) = 0  (gauge condition)
    """
    rho_hat = da.fft.fft2(to_dask(n) - 1.0)

    KX, KY = np.meshgrid(grid.kx, grid.ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0              # avoid /0; DC mode zeroed by mask below

    mask = np.ones_like(k2)
    mask[0, 0] = 0.0            # gauge: phi_hat(0,0) = 0

    phi_hat = rho_hat / to_dask(k2) * to_dask(mask)
    return da.real(da.fft.ifft2(phi_hat))


def electric_field_from_potential(phi, grid):
    """Compute E = -nabla(phi) via spectral differentiation using da.fft.fft2.

    In Fourier space:
      E_hat_x(k) = -1j * kx * phi_hat(k)
      E_hat_y(k) = -1j * ky * phi_hat(k)
    Returns array of shape (Nx, Ny, 2).
    """
    phi_hat = da.fft.fft2(to_dask(phi))
    KX, KY = np.meshgrid(grid.kx, grid.ky, indexing="ij")
    Ex = da.real(da.fft.ifft2(-1j * to_dask(KX) * phi_hat))
    Ey = da.real(da.fft.ifft2(-1j * to_dask(KY) * phi_hat))
    return da.stack([Ex, Ey], axis=-1)


def electric_field_energy(Efield, grid):
    return 0.5 * da.sum(to_dask(Efield) ** 2) * grid.dV_2D


def measure(cfg, f, Efield, it, t_actual):
    """Compute and append conserved-variable diagnostics for a 4D f[x, y, vx, vy]."""
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    diag_file_path = data_dir / "diagnostics.csv"

    f_da = to_dask(f)
    if f_da.ndim != 4:
        raise ValueError(
            f"Padawan, the electric force is not with you -- "
            f"f must have 4 dimensions, not {f_da.ndim}."
        )

    grid  = cfg.grid
    vx_bc = to_dask(grid.vx)[(None, None, slice(None), None)]
    vy_bc = to_dask(grid.vy)[(None, None, None, slice(None))]
    v2    = vx_bc ** 2 + vy_bc ** 2

    epot       = float(electric_field_energy(Efield, grid).compute())
    ekin       = float((0.5 * da.sum(f_da * v2) * grid.dV_4D).compute())
    l2norm_sq  = float((da.sum(f_da ** 2) * grid.dV_4D).compute())
    mass       = float((da.sum(f_da) * grid.dV_4D).compute())
    momentum_x = float((da.sum(f_da * vx_bc) * grid.dV_4D).compute())
    momentum_y = float((da.sum(f_da * vy_bc) * grid.dV_4D).compute())

    data = {
        "iter":       it,
        "time":       float(t_actual),
        "ekin":       ekin,
        "epot":       epot,
        "etot":       ekin + epot,
        "l2norm":     float(np.sqrt(l2norm_sq)),
        "mass":       mass,
        "momentum":   float(np.hypot(momentum_x, momentum_y)),
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }

    reset = it == 0
    write_header = reset or not diag_file_path.is_file()
    with open(diag_file_path, mode="w" if reset else "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(data)


# ---------------------------------------------------------------------------
# Deisa analytics callbacks - receive data assembled from all MPI ranks
# ---------------------------------------------------------------------------

deisa = Deisa()


@deisa.register("fdistribu")
def compute_diagnostics(fdistribu_chunks):
    client = get_client()
    coords = client.gather(client.get_dataset("coords"))

    if _MEASURE_CFG is None:
        init_measure_config(
            x  = np.array(coords['MeshX']),
            y  = np.array(coords['MeshY']),
            vx = np.array(coords['MeshVx']),
            vy = np.array(coords['MeshVy']),
        )

    fdistribu = np.array(fdistribu_chunks[0])  # (Nsp, Nx, Ny, Nvx, Nvy)
    t_actual  = float(np.array(coords['absolute_time'])[0])
    timestep  = int(fdistribu_chunks[0].t)

    cfg = get_measure_config()

    n      = density(fdistribu, cfg.grid)
    phi    = poisson_fft(n, cfg.grid)
    Efield = electric_field_from_potential(phi, cfg.grid)

    Nsp = fdistribu.shape[0]
    for isp in range(Nsp):
        data_dir = cfg.paths.data_dir if Nsp == 1 else cfg.paths.data_dir / f"species_{isp}"
        sp_cfg = Config(paths=PathsConfig(data_dir), grid=cfg.grid)
        measure(sp_cfg, fdistribu[isp], Efield, timestep, t_actual)


deisa.execute_callbacks()
