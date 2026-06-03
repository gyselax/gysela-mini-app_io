import os
import sys
import dask
import h5py
import json
import yaml
import time
import logging
import dask.array as da
from deisa.ray import Deisa
from distributed import Variable, Queue, get_client

logging.basicConfig(level=logging.DEBUG)

gys_io_config = sys.argv[1]
with open(gys_io_config, 'r') as config_file:
    nb_iter = yaml.safe_load(config_file)["Application"]["n_iterations"]

print(f"[Deisa] Start connection of Deisa \n", flush=True)
deisa = Deisa()
print("[Deisa] Connected\n", flush=True)

@deisa.register("density", "velocity", "temperature")
def sum_moments(density, velocity, temperature):
    sum_density = density[0].sum().compute() 
    sum_velocity = velocity[0].sum().compute() 
    sum_temperature = temperature[0].sum().compute() 

    print(f"[Deisa] Iteration {density[0].t}", flush=True)
    print(f"[Deisa] sum density {sum_density}", flush=True)
    print(f"[Deisa] sum velocity {sum_velocity}", flush=True)
    print(f"[Deisa] sum temperature {sum_temperature}", flush=True)

    if density[0].t == nb_iter-1:
        with h5py.File("deisa_fluid_moments.h5", 'w') as f:
            f.create_dataset("density", data=density[0])
            f.create_dataset("mean_velocity", data=velocity[0])
            f.create_dataset("temperature", data=temperature[0])

deisa.execute_callbacks()

