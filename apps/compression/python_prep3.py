# class Mesh:
from mpi4py import MPI
import pdi
import h5py
import numpy as np

def print_mesh_info(meshx_py, meshy_py, meshvx_py, meshvy_py, local_fdistribu_extents_py, local_fdistribu_starts_py) :
    # construir le mesh 
    print("========================calling print_mesh_info")
    
    print("\n===================meshvx_py = ", meshvx_py)

    

def prep4catalyst(fdistribu_py, electrostatic_potential_py, iter_py, time_py, meshx_py, meshy_py, meshvx_py, meshvy_py, local_fdistribu_extents_py, local_fdistribu_starts_py):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank ==0: 
        print("========================calling prep4catalyst at iteration ", iter_py)

    # print("\n=================== rank ", rank, ": local_fdistribu_extents_py = ", local_fdistribu_extents_py)

    # print("\n=================== rank ", rank, ": local_fdistribu_starts_py = ", local_fdistribu_starts_py)

    PT_Y=int(local_fdistribu_extents_py[2]/2)
    PT_VY=int(local_fdistribu_extents_py[4]/2)

    fdistribu_selected=fdistribu_py[0, :, PT_Y, :, PT_VY].copy()
    
    coord_X = meshvx_py[local_fdistribu_starts_py[3]:local_fdistribu_starts_py[3]+local_fdistribu_extents_py[3]]

    # print("\n=================== rank ", rank, ": coord_X = ", coord_X)
    # coordonnee dans le champs de vitesse

    # print("\n=================== rank ", rank, ": selected = ", fdistribu_selected)
        # print("rank ", rank, ": Coord_X = ", coord_X)
    if iter_py%10 == 0:
        pdi.multi_expose("catalyst_execute", [
                    ("cycle", iter_py, pdi.OUT),
                    ("time", time_py, pdi.OUT),
                    ("bidon", fdistribu_selected, pdi.OUT),
                    ("local_points_X", meshx_py, pdi.OUT),
                    ("local_points_Y", coord_X, pdi.OUT),
                    ])
