import os
import sys
import dask
import h5py
import json
import yaml
import time
import logging
import xarray as xr
import dask.array as da
from pathlib import Path
from deisa.dask import Deisa
from distributed import Variable, Queue, get_client

#logging.basicConfig(level=logging.DEBUG)

deisa = Deisa()

@deisa.register("partial_density", 
        "partial_velocity", 
        "partial_temperature")
def sum_moments(partial_density, partial_velocity, partial_temperature):
    # ===== will no longer be needed once xarrays can be sent through the bridge ======
    if len(partial_density[0].shape) == 6:
        dims = ('species', 'tor1', 'tor2', 'tor3', 'vpar', 'mu')
    elif len(partial_density[0].shape) == 4:
        dims = ('species', 'tor1', 'tor2', 'tor3')

    timestep = partial_density[0].t

    density = xr.DataArray(
            partial_density[0],
            dims=dims
            )
    velocity = xr.DataArray(
            partial_velocity[0],
            dims=dims
            )
    temperature = xr.DataArray(
            partial_temperature[0],
            dims=dims
            )
    # =================================================================================

    if 'vpar' in density.dims and 'mu' in density.dims:
        density = density.sum(dim=('vpar', 'mu'))
        velocity = velocity.sum(dim=('vpar', 'mu'))
        temperature = temperature.sum(dim=('vpar', 'mu'))

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f'output/optimised_fluid_moments_{timestep}.h5', 'w') as fh5:
        fh5.create_dataset('density', data=density)
        fh5.create_dataset('velocity', data=velocity)
        fh5.create_dataset('temperature', data=temperature)

deisa.execute_callbacks()

