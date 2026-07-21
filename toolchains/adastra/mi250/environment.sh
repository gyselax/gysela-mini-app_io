#!/bin/bash

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "This script must be sourced not executed."
    echo ". $0"
    exit 1
fi

module purge

SPACK_USER_VERSION="spack-user-5.0.0"

export SPACK_USER_PREFIX="/lus/work/CT5/gen2224/SHARED/gysela-mini-app-GENOA"
export SPACK_USER_CACHE_PATH="${SPACK_USER_PREFIX}/cache"

export SPACK_USER_CONFIG_PATH="${ALL_CCFRWORK}/gyselalibxx-spack-install-py314-genoa/configuration"

# Avoid too many temporary files in the Spack installation tree
export PYTHONPYCACHEPREFIX=$ALL_CCFRSCRATCH/pycache

module load develop "${SPACK_USER_VERSION}"
module load llvm/20.1.6
which spack
spack debug report

. /lus/home/softs/gaia/prod/5.0.0/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_plac/spack-1.0.1-gcc-13.2.1-3ra4/share/spack/setup-env.sh

# Activate the environment in the scratch directory
# spack env activate $SPACK_USER_PREFIX

eval -- "$(
    spack \
        --env "$SPACK_USER_PREFIX" \
        load --sh \
        cmake \
        ddc \
        gcc \
        ginkgo \
        googletest \
        kokkos \
        kokkos-fft \
        kokkos-kernels \
        kokkos-tools \
        koliop \
        lapack \
        mpi \
        ninja \
        paraconf \
        pdi \
        pdiplugin-decl-hdf5 \
        pdiplugin-decl-netcdf \
        pdiplugin-mpi \
        pdiplugin-set-value \
        pdiplugin-trace \
        pdiplugin-pycall \
        python \
        hdf5 \
        arrow \
        py-dask \
        py-dask-ml \
        py-deisa-dask \
        py-h5py \
        py-imageio \
        py-matplotlib \
        py-netcdf4 \
        py-numpy \
        py-scipy \
        py-sympy \
        py-xarray \
        py-pyyaml
)"

# Add Kokkos Tools to the `LD_LIBRARY_PATH`
export LD_LIBRARY_PATH="$(spack --env "$SPACK_USER_PREFIX" location -i kokkos-tools)/lib64:$LD_LIBRARY_PATH"

export GYSELALIBXX_OPENBLAS_ROOT="$(spack --env "$SPACK_USER_PREFIX" location -i openblas)"
export PYTHONPATH="$(spack --env "$SPACK_USER_PREFIX" location -i pdi)/lib/python3.13/site-packages:$PYTHONPATH"
