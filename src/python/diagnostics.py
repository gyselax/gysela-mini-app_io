"""In-situ diagnostics: conserved-variable calculations and deisa analytics callback."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
import numpy as np
from deisa.dask import Deisa
from distributed import get_client

import compression_diagnostics

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
        Nx = da.asarray(self.x).size
        Ny = da.asarray(self.y).size
        self.kx = 2.0 * da.pi * da.fft.fftfreq(Nx, d=self.dx)
        self.ky = 2.0 * da.pi * da.fft.fftfreq(Ny, d=self.dy)

    @staticmethod
    def spacing(coord):
        values = da.asarray(coord)
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
    return da.sum(fdistribu, axis=(0, 3, 4)) * grid.dvx * grid.dvy


def poisson_fft(n, grid):
    """Solve Laplacian(phi) = -(n - 1) on a doubly-periodic domain using da.fft.fft2.

    In Fourier space: -(kx^2 + ky^2) * phi_hat(k) = rho_hat(k)
    phi_hat(0,0) = 0  (gauge condition)
    """
    rho_hat = da.fft.fft2(n - 1.0)

    KX, KY = da.meshgrid(grid.kx, grid.ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0              # avoid /0; DC mode zeroed by mask below

    mask = da.ones_like(k2)
    mask[0, 0] = 0.0            # gauge: phi_hat(0,0) = 0

    phi_hat = rho_hat / k2 * mask
    return da.real(da.fft.ifft2(phi_hat))


def electric_field_from_potential(phi, grid):
    """Compute E = -nabla(phi) via spectral differentiation using da.fft.fft2.

    In Fourier space:
      E_hat_x(k) = -1j * kx * phi_hat(k)
      E_hat_y(k) = -1j * ky * phi_hat(k)
    Returns array of shape (Nx, Ny, 2).
    """
    phi_hat = da.fft.fft2(phi)
    KX, KY = da.meshgrid(grid.kx, grid.ky, indexing="ij")
    Ex = da.real(da.fft.ifft2(-1j * KX * phi_hat))
    Ey = da.real(da.fft.ifft2(-1j * KY * phi_hat))
    return da.stack([Ex, Ey], axis=-1)


def electric_field_energy(Efield, grid):
    return 0.5 * da.sum(Efield ** 2) * grid.dV_2D

_INITIALIZED_DIAG_FILES = set()

def measure(cfg, f, Efield, it, t_actual):
    """Compute and append conserved-variable diagnostics for a 4D f[x, y, vx, vy]."""
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    diag_file_path = data_dir / "diagnostics.csv"

    if f.ndim != 4:
        raise ValueError(
            f"Padawan, the electric force is not with you -- "
            f"f must have 4 dimensions, not {f.ndim}."
        )

    grid  = cfg.grid
    vx_bc = grid.vx[(None, None, slice(None), None)]
    vy_bc = grid.vy[(None, None, None, slice(None))]
    v2    = vx_bc ** 2 + vy_bc ** 2

    epot       = electric_field_energy(Efield, grid)
    ekin       = 0.5 * da.sum(f * v2) * grid.dV_4D
    l2norm_sq  = da.sum(f ** 2) * grid.dV_4D
    mass       = da.sum(f) * grid.dV_4D
    momentum_x = da.sum(f * vx_bc) * grid.dV_4D
    momentum_y = da.sum(f * vy_bc) * grid.dV_4D

    client = get_client()
    futures = client.compute([
        epot, ekin, l2norm_sq, mass, momentum_x, momentum_y
    ])
    epot, ekin, l2norm_sq, mass, momentum_x, momentum_y = map(
        float, deisa.client.gather(futures)
    )

    data = {
        "iter":       it,
        "time":       float(t_actual),
        "ekin":       ekin,
        "epot":       epot,
        "etot":       ekin + epot,
        "l2norm":     float(da.sqrt(l2norm_sq)),
        "mass":       mass,
        "momentum":   float(da.hypot(momentum_x, momentum_y)),
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }

    reset = diag_file_path not in _INITIALIZED_DIAG_FILES
    _INITIALIZED_DIAG_FILES.add(diag_file_path)
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

@deisa.register("fdistribu_offline")
def compute_offline_compression(fdistribu_chunks):
    timestep = int(fdistribu_chunks[0].t)
    fdistribu_global = np.array(fdistribu_chunks[0])

    compression_diagnostics.run_offline_compression_on_global_array(fdistribu_global, timestep)

    deisa.set("fdistribu_offline_done", True, timestep=timestep)


@deisa.register("fdistribu", "absolute_time", "deltat", "MeshX", "MeshY", "MeshVx", "MeshVy")
def compute_diagnostics(fdistribu, time, deltat, mx, my, mvx, mvy):

    if _MEASURE_CFG is None:
        init_measure_config(
            x  = mx[0],
            y  = my[0],
            vx = mvx[0],
            vy = mvy[0],
        )

    # This caused a race condition conflict on persee (sim runs faster then the diagnostics
    # and republishes t_actual before the one attached to the fdistribu chunk could be used)
    # t_actual  = float(np.array(coords['absolute_time'])[0])
    timestep  = int(fdistribu[0].t)
    t_actual = timestep * float(np.array(deltat[0]).reshape(-1)[0])

    cfg = get_measure_config()

    n      = density(fdistribu[0], cfg.grid)
    phi    = poisson_fft(n, cfg.grid)
    Efield = electric_field_from_potential(phi, cfg.grid)

    Nsp = fdistribu[0].shape[0]
    for isp in range(Nsp):
        data_dir = cfg.paths.data_dir if Nsp == 1 else cfg.paths.data_dir / f"species_{isp}"
        sp_cfg = Config(paths=PathsConfig(data_dir), grid=cfg.grid)
        measure(sp_cfg, fdistribu[0][isp], Efield, timestep, t_actual)


deisa.execute_callbacks()

def _sort_diagnostics_files():
    """Rewrite each diagnostics.csv sorted by iter (rows are appended in
    task-completion order, which is not guaranteed to be time order)."""
    for path in _INITIALIZED_DIAG_FILES:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = sorted(reader, key=lambda r: int(r["iter"]))
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


_sort_diagnostics_files()
