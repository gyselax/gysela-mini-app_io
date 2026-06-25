SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd $SCRIPT_DIR/../.. && pwd)"

source $BASE_DIR/toolchains/persee/xeon/environment.sh

# spack env activate warns about a missing cloudpickle repo and skips loading the
# runtime environment (PYTHONPATH), so pdi and deisa Python packages are not visible.
# Add them explicitly via the stable view path.
export PYTHONPATH=/data/gyselarunner/gysela-io-env-deisa/.spack-env/view/lib/python3.13/site-packages:${PYTHONPATH:-}

. $BASE_DIR/.gys_env/bin/activate
