#!/bin/bash

SIMU_NODES=${1:-1}
DASK_WORKERS=${2:-1}
DASK_THREADS_PER_WORKER=${3:-2}

#DASK_WORKER_MEMORY=8589934592
#DASK_WORKER_MEMORY="8GiB"
#DASK_WORKER_MEMORY=4294967296
#DASK_WORKER_MEMORY=3221225472
DASK_WORKER_MEMORY=2147483648
#DASK_WORKER_MEMORY=1610612736
#DASK_WORKER_MEMORY=1073741824
#DASK_WORKER_MEMORY=536870912

#SCHEFILE=~/gysela-mini-app_io/scheduler.json

. ~/env-miniapp-gysela.sh
set -x
#export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=forkserver
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=False

NODES=($(sort -u $OAR_NODEFILE))
WORKER_NODES=(${NODES[@]:0:${DASK_WORKERS}})
MPI_NODES=(${NODES[@]:${DASK_WORKERS}:${SIMU_NODES}})
MPI_NODEFILE=$(mktemp)
printf "%s\n" "${MPI_NODES[@]}" > $MPI_NODEFILE

cd ~/gysela-mini-app_io
#rm -f $SCHEFILE
rm -rf gysela_plots/[dhn]*/*

#echo "Launch scheduler"
#dask scheduler --scheduler-file=$SCHEFILE &
#dask_sch_pid=$!

source ~/venv/bin/activate

PORT=4242
HEAD_ADDRESS=${WORKER_NODES}:$PORT
oarsh ${WORKER_NODES} ". ~/env-miniapp-gysela.sh && \
  source ~/venv/bin/activate && \
  ray start --head --port=$PORT --block" &
echo "Head node started"

sleep 10
oarsh ${WORKER_NODES} ". ~/env-miniapp-gysela.sh && \
  source ~/venv/bin/activate && \

  echo 'Launch analytics'
  python3 ~/gysela-mini-app_io/python/analytics.py  ~/gysela-mini-app_io/apps/gys_io.yaml " &
analytics_pid=$!
echo "Analytics started"

echo "Launch workers"
for NODE in "${MPI_NODES[@]}"; do
    oarsh ${NODE} ". ~/env-miniapp-gysela.sh && \
      source ~/venv/bin/activate && \
      ray start --address ${HEAD_ADDRESS} --block" &
done
sleep 10

echo "Launch simulation"
mpirun -machinefile $MPI_NODEFILE \
	--prefix $(dirname $(dirname $(which mpirun))) \
	-x LD_LIBRARY_PATH \
	-x PYTHONPATH \
	-x PDI_PLUGIN_PATH \
	-n $SIMU_NODES bash -c "source ~/venv/bin/activate && ~/gysela-mini-app_io/build/apps/gys_io  ~/gysela-mini-app_io/apps/gys_io.yaml  ~/gysela-mini-app_io/apps/pdi_default.yaml " &
simu_pid=$!

echo "Simulation started"

wait ${analytics_pid} 
echo "Analytics over"

wait ${simu_pid}
echo "Simulation over"

rm -f $MPI_NODEFILE
