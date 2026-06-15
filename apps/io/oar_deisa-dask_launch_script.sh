#!/bin/bash

SIMU_NODES=${1:-1}
DASK_WORKERS=${2:-1}
DASK_THREADS_PER_WORKER=${3:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd $SCRIPT_DIR/../.. && pwd)"

. $SCRIPT_DIR/env-miniapp-io.sh

export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=False

NODES=($(sort -u $OAR_NODEFILE))
WORKER_NODES=(${NODES[@]:0:${DASK_WORKERS}})
MPI_NODES=(${NODES[@]:${DASK_WORKERS}:${SIMU_NODES}})
MPI_NODEFILE=$(mktemp)
printf "%s\n" "${MPI_NODES[@]}" > $MPI_NODEFILE

cd $BASE_DIR

SCHEFILE="$BASE_DIR/scheduler.json"
rm -f $SCHEFILE

echo "Launch scheduler"
dask scheduler --scheduler-file=$SCHEFILE &
dask_sch_pid=$!
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo -n .
done

export DEISA_DASK_SCHEDULER_ADDRESS=$(jq -r '.["address"]' $SCHEFILE)

echo "Launch workers"
dask_worker_pids=()
for NODE in "${WORKER_NODES[@]}"; do
    oarsh ${NODE} ". $BASE_DIR/apps/io/env-miniapp-io.sh && dask worker \
        --nworkers 1 \
        --nthreads ${DASK_THREADS_PER_WORKER} \
        --local-directory /tmp \
        --scheduler-file=${SCHEFILE}" &
    dask_worker_pids+=($!)
done
sleep 10

echo "Launch analytics"
python3 src/python/analytics.py &
analytics_pid=$!

echo "Launch simulation"
mpirun -machinefile $MPI_NODEFILE \
	--prefix $(dirname $(dirname $(which mpirun))) \
	-x PYTHONPATH \
	-x DEISA_DASK_SCHEDULER_ADDRESS \
	-n $SIMU_NODES build/apps/io/gys_io apps/io/gys_io.yaml apps/io/pdi_deisa.yaml &
simu_pid=$!

wait ${analytics_pid}
echo "Analytics over"

wait ${simu_pid}
echo "Simulation over"

rm -f $MPI_NODEFILE

for NODE in "${WORKER_NODES[@]}"; do
    oarsh ${NODE} "pkill -9 -f 'dask worker'" &
done

kill -9 ${dask_sch_pid}
