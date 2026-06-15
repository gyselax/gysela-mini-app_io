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
from fluid_moments import FluidMoments
from deisa.dask import Deisa
from distributed import Variable, Queue, get_client

#logging.basicConfig(level=logging.DEBUG)

deisa = Deisa()

@deisa.register("fdistribu_raw")
def compute_moments(fdistribu_raw):
    # ===== will no longer be needed once xarrays can be sent through the bridge ======
    client = get_client()
    coords = client.gather(client.get_dataset("coords"))

    fdistribu = xr.DataArray(
            fdistribu_raw[0],
            dims=('species', 'tor1', 'tor2', 'tor3', 'vpar', 'mu'),
            coords={
                'species': [0, 1],
                'tor1': coords['tor1'],
                'tor2': coords['tor2'],
                'tor3': coords['tor3'],
                'vpar': coords['vpar'],
                'mu':   coords['mu'],
            },
    )
    # =================================================================================

    timestep = fdistribu_raw[0].t

    fm = FluidMoments(coords["vpar"], coords["mu"])

    density = fm.compute_density(fdistribu)
    velocity = fm.compute_velocity(fdistribu, density)
    temperature = fm.compute_temperature(fdistribu, density, velocity)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f'output/normal_fluid_moments_{timestep}.h5', 'w') as fh5:
        fh5.create_dataset('density', data=density)
        fh5.create_dataset('velocity', data=velocity)
        fh5.create_dataset('temperature', data=temperature)

deisa.execute_callbacks()

