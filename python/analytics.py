import os
import sys
import dask
import h5py
import json
import yaml
import time
import logging
import dask.array as da
from deisa.dask import Deisa
from distributed import Variable, Queue, get_client

logging.basicConfig(level=logging.DEBUG)

gys_io_config = sys.argv[1]
with open(gys_io_config, 'r') as config_file:
    nb_iter = yaml.safe_load(config_file)["Application"]["n_iterations"]

with open("scheduler.json", 'r') as scheduler_file:
    dask_addr = json.load(scheduler_file)["address"]

print(f"[Deisa] Start connection of Deisa to {dask_addr}\n")
deisa = Deisa()
print("[Deisa] Connected\n")

@deisa.register("density", "velocity", "temperature")
def sum_moments(density, velocity, temperature):
    sum_density = density[0].sum().compute() 
    sum_velocity = velocity[0].sum().compute() 
    sum_temperature = temperature[0].sum().compute() 

    print(f"[Deisa] Iteration {density[0].t}")
    print(f"[Deisa] sum density {sum_density}\n")
    print(f"[Deisa] sum velocity {sum_velocity}\n")
    print(f"[Deisa] sum temperature {sum_temperature}\n")

    if density[0].t == nb_iter-1:
        with h5py.File("deisa_fluid_moments.h5", 'w') as f:
            f.create_dataset("density", data=density[0])
            f.create_dataset("mean_velocity", data=velocity[0])
            f.create_dataset("temperature", data=temperature[0])

deisa.execute_callbacks()

