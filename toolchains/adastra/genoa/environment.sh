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

# Avoid too many temporary files in the Spack installation tree
export PYTHONPYCACHEPREFIX=$ALL_CCFRSCRATCH/pycache

module load develop "${SPACK_USER_VERSION}"
module load llvm/20.1.6
which spack
spack debug report
# Spack must work in a clean, purged environment so it can load modules without
# having to purge itself or clearing environment variables (which it does not
# do..). When we spack env activate, the same constraint applies.
# Use spack load instead of an environment activation as it should limit the
# inode produced by the environment's view.
# eval -- "$(spack env activate --prompt --sh gyselalibxx-spack-environment)"
# unalias despacktivate
# unset despacktivate
# function despacktivate() {
#     eval "$(spack env deactivate --sh)"
# }

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
