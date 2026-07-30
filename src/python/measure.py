"""Simulation diagnostics for 4D distribution functions using Dask."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np

ArrayLike = Union[np.ndarray, da.Array]

_MEASURE_CFG: Optional["Config"] = None


@dataclass
class PathsConfig:
    data_dir: Path


@dataclass
class GridConfig:
    """Uniform 2D2V mesh (x, y, vx, vy)."""

    x: ArrayLike
    y: ArrayLike
    vx: ArrayLike
    vy: ArrayLike

    @staticmethod
    def _spacing(coord: ArrayLike) -> float:
        values = np.asarray(coord)
        if values.size < 2:
            raise ValueError("Mesh coordinate must contain at least two points.")
        return float(values[1] - values[0])

    @property
    def dx(self) -> float:
        return self._spacing(self.x)

    @property
    def dy(self) -> float:
        return self._spacing(self.y)

    @property
    def dvx(self) -> float:
        return self._spacing(self.vx)

    @property
    def dvy(self) -> float:
        return self._spacing(self.vy)

    @property
    def dV_2D(self) -> float:
        return self.dx * self.dy

    @property
    def dV_4D(self) -> float:
        return self.dx * self.dy * self.dvx * self.dvy


@dataclass
class Config:
    paths: PathsConfig
    grid: GridConfig


def _get_mpi_comm() -> Any:
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError:
        return None


def to_dask_array(array: ArrayLike) -> da.Array:
    return array if isinstance(array, da.Array) else da.from_array(array)


def init_measure_config(
    x: ArrayLike,
    y: ArrayLike,
    vx: ArrayLike,
    vy: ArrayLike,
    data_dir: Union[str, Path] = ".",
) -> Config:
    """Store the mesh configuration used by in-situ PDI measurements."""
    global _MEASURE_CFG
    _MEASURE_CFG = Config(
        PathsConfig(Path(data_dir)),
        GridConfig(x, y, vx, vy),
    )
    return _MEASURE_CFG


def get_measure_config() -> Config:
    if _MEASURE_CFG is None:
        raise RuntimeError("measure config not initialized; call init_measure_config first.")
    return _MEASURE_CFG


def _broadcast_vx(f: da.Array, vx: ArrayLike) -> da.Array:
    vx_da = to_dask_array(vx)
    return vx_da[(None, None, slice(None), None)]


def _broadcast_vy(f: da.Array, vy: ArrayLike) -> da.Array:
    vy_da = to_dask_array(vy)
    return vy_da[(None, None, None, slice(None))]


def _velocity_squared(f: da.Array, grid: GridConfig) -> da.Array:
    vx = _broadcast_vx(f, grid.vx)
    vy = _broadcast_vy(f, grid.vy)
    return vx * vx + vy * vy


def electric_field_from_potential(phi: ArrayLike, grid: GridConfig) -> da.Array:
    """Build (x, y, 2) electric field components from electrostatic potential."""
    phi_da = to_dask_array(phi)
    dphi_dx = (da.roll(phi_da, -1, axis=0) - da.roll(phi_da, 1, axis=0)) / (2.0 * grid.dx)
    dphi_dy = (da.roll(phi_da, -1, axis=1) - da.roll(phi_da, 1, axis=1)) / (2.0 * grid.dy)
    return da.stack([-dphi_dx, -dphi_dy], axis=-1)


def _electric_field_energy(Efield: da.Array, grid: GridConfig) -> da.Array:
    if Efield.ndim == 2:
        return 0.5 * da.sum(Efield * Efield) * grid.dV_2D
    if Efield.ndim == 3 and Efield.shape[-1] == 2:
        return 0.5 * da.sum(Efield * Efield) * grid.dV_2D
    if Efield.ndim == 3 and Efield.shape[0] == 2:
        return 0.5 * da.sum(Efield * Efield) * grid.dV_2D
    raise ValueError(
        "Efield must have shape (Nx, Ny), (Nx, Ny, 2), or (2, Nx, Ny). "
        f"Got {Efield.shape}."
    )


def _reduce_scalar(value: float, comm: Any) -> float:
    if comm is None:
        return value
    from mpi4py import MPI

    return float(comm.allreduce(value, op=MPI.SUM))


def measure(
    cfg: Config,
    f: ArrayLike,
    Efield: ArrayLike,
    it: int,
    t_actual: float,
    comm: Any = None,
) -> None:
    """Calculate and append physical diagnostics for a 4D distribution f[x, y, vx, vy]."""
    if comm is None:
        comm = _get_mpi_comm()

    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    diag_file_path = data_dir / "diagnostics.csv"

    f_da = to_dask_array(f)
    E_da = to_dask_array(Efield)

    if f_da.ndim != 4:
        raise ValueError(f"Expected 4D distribution f[x, y, vx, vy], got shape {f_da.shape}.")

    grid = cfg.grid
    dV_4D = grid.dV_4D
    v2 = _velocity_squared(f_da, grid)
    vx = _broadcast_vx(f_da, grid.vx)
    vy = _broadcast_vy(f_da, grid.vy)

    epot = _reduce_scalar(float(_electric_field_energy(E_da, grid).compute()), comm)
    ekin = _reduce_scalar(float((0.5 * da.sum(f_da * v2) * dV_4D).compute()), comm)
    etot = ekin + epot

    l2norm_sq = _reduce_scalar(float((da.sum(f_da * f_da) * dV_4D).compute()), comm)
    l2norm = float(np.sqrt(l2norm_sq))
    mass = _reduce_scalar(float((da.sum(f_da) * dV_4D).compute()), comm)
    momentum_x = _reduce_scalar(float((da.sum(f_da * vx) * dV_4D).compute()), comm)
    momentum_y = _reduce_scalar(float((da.sum(f_da * vy) * dV_4D).compute()), comm)
    momentum = float(np.hypot(momentum_x, momentum_y))

    if comm is not None and comm.Get_rank() != 0:
        return

    data = {
        "iter": it,
        "time": float(t_actual),
        "ekin": ekin,
        "epot": epot,
        "etot": etot,
        "l2norm": l2norm,
        "mass": mass,
        "momentum": momentum,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }

    reset = it == 0
    write_header = reset or not diag_file_path.is_file()
    with open(diag_file_path, mode="w" if reset else "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def run_measure_insitu(
    fdistribu: ArrayLike,
    phi: ArrayLike,
    it: int,
    t_actual: float,
    iter_offset: int = 0,
    species: int = 0,
    data_dir: Union[str, Path] = ".",
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    vx: Optional[ArrayLike] = None,
    vy: Optional[ArrayLike] = None,
) -> None:
    """Convenience entry point for PDI pycall events."""
    try:
        cfg = get_measure_config()
    except RuntimeError:
        if x is None or y is None or vx is None or vy is None:
            raise RuntimeError(
                "measure config not initialized and mesh coordinates were not provided."
            ) from None
        cfg = init_measure_config(x, y, vx, vy, data_dir=data_dir)

    f = np.asarray(fdistribu)[species, ...]
    Efield = electric_field_from_potential(phi, cfg.grid)
    measure(cfg, f, Efield, it + iter_offset, t_actual)
