#!/usr/bin/env bash
# Source the Gyselalib++ toolchain, install Python deps, build, and run tests.
#
# Usage:
#   ./installer.sh <MACHINE>
#
# Examples:
#   ./installer.sh persee/xeon
#   ./installer.sh h100.jean-zay.spack

set -eo pipefail

MACHINE="${1:-${GYSELA_MACHINE:-}}"
if [[ -z "${MACHINE}" ]]; then
  host="$(hostname -s 2>/dev/null || hostname)"
  case "${host}" in
    persee*) MACHINE="persee/xeon" ;;
  esac
fi

if [[ -z "${MACHINE}" ]]; then
  echo "Usage: $0 <MACHINE>" >&2
  echo "Example: $0 persee/xeon" >&2
  echo "Or set GYSELA_MACHINE=persee/xeon" >&2
  echo "Toolchains: src/external/gyselalibxx/toolchains/" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TOOLCHAIN_DIR="${REPO_ROOT}/src/external/gyselalibxx/toolchains/${MACHINE}"
ENV_SH="${TOOLCHAIN_DIR}/environment.sh"
TOOLCHAIN_CMAKE="${TOOLCHAIN_DIR}/toolchain.cmake"

for f in "${ENV_SH}" "${TOOLCHAIN_CMAKE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing: ${f}" >&2
    exit 1
  fi
done

echo "==> Toolchain: ${MACHINE}"
echo "==> Sourcing ${ENV_SH}"
# shellcheck source=/dev/null
source "${ENV_SH}"

export PYTHONPATH="${REPO_ROOT}/src/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> pip install -e .[dev]"
python -m venv venv
source ./venv/bin/activate
python -m pip install -e ".[dev]"

BUILD_DIR="${REPO_ROOT}/build"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

echo "==> CMake configure"
cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_CMAKE}"

echo "==> CMake build (-j ${JOBS})"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo "==> pytest"
pytest

echo "==> Done"
