# class Mesh:
from mpi4py import MPI
import pdi

def print_mesh_info(meshx_py, meshy_py, meshvx_py, meshvy_py, local_fdistribu_extents_py, local_fdistribu_starts_py) :
    # construir le mesh 
    print("========================calling print_mesh_info")
    
    # print("\n===================meshx_py = ", meshx_py)

    # print("\n===================meshy_py = ", meshy_py)

    print("\n===================meshvx_py = ", meshvx_py)

    # print("\n===================meshvy_py = ", meshvy_py)

    

def prep4catalyst(fdistribu_py, electrostatic_potential_py, iter_py, meshx_py, meshy_py, meshvx_py, meshvy_py, local_fdistribu_extents_py, local_fdistribu_starts_py):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("========================calling prep4catalyst")

    print("\n=================== rank ", rank, ": local_fdistribu_extents_py = ", local_fdistribu_extents_py)

    print("\n=================== rank ", rank, ": local_fdistribu_starts_py = ", local_fdistribu_starts_py)

    time = iter_py * 0.1


    PT_Y=int(local_fdistribu_extents_py[2]/2)
    PT_VY=int(local_fdistribu_extents_py[4]/2)

    fdistribu_selected=fdistribu_py[0, :, PT_Y, :, PT_VY]
    coord_X = meshvx_py[local_fdistribu_starts_py[3]:local_fdistribu_starts_py[3]+local_fdistribu_extents_py[3]]
    # coordonnee dans le champs de vitesse


        # print("rank ", rank, ": Coord_X = ", coord_X)
    pdi.multi_expose("catalyst_execute", [
                    ("cycle", iter_py, pdi.OUT),
                    ("time", time, pdi.OUT),
                    ("bidon", fdistribu_selected, pdi.OUT),
                    ("local_points_X", meshx_py, pdi.OUT),
                    ("local_points_Y", coord_X, pdi.OUT),
                    # ("bidon_electro_prot", electrostatic_potential_py, pdi.OUT),
                    ])
    



    