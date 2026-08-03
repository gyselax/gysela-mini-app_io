#!/bin/bash

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "This script must be sourced not executed."
    echo ". $0"
    exit 1
fi

module purge

if command -v spack >/dev/null 2>&1
then
    spack env deactivate
else
    . /data/gyselarunner/spack-1.1.1/share/spack/setup-env.sh
fi

# The hdf5 package is injecting the environment view `lib` path to `LD_LIBRARY_PATH`
# which causes spurious segfaults for system executables, we manually remove it.
LD_LIBRARY_PATH_TMP="$LD_LIBRARY_PATH"
spack env activate /data/gyselarunner/gysela-io-env-deisa-cuda/
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_TMP"
unset LD_LIBRARY_PATH_TMP

module load gcc/13
# Looks suspicious to set LD_PRELOAD, so we unset it
unset LD_PRELOAD

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

# Add Kokkos Tools to the `LD_LIBRARY_PATH`
export LD_LIBRARY_PATH="$(spack location -i kokkos-tools)/lib64:$LD_LIBRARY_PATH"

export PYTHONPATH=/data/gyselarunner/gysela-io-env-deisa-cuda/.spack-env/view/lib/python3.13/site-packages:${PYTHONPATH:-}

# Every MPI rank embeds its own Python/JAX runtime (via PDI's pycall plugin),
# and JAX's default behavior is to preallocate ~75% of GPU memory on first
# use, per process. With several ranks sharing one GPU on this node, that
# alone exhausts the device before any compression work starts. Switch JAX to
# on-demand allocation instead.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
