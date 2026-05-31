#!/bin/bash

SIMU_NODES=${1:-1}
DASK_WORKERS=${2:-1}

SCHEFILE=scheduler.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -f $SCHEFILE

echo "Launch scheduler"
dask scheduler --scheduler-file=$SCHEFILE &
dask_sch_pid=$!

while ! [ -f $SCHEFILE ]; do
	sleep 1
	echo -n .
done

echo "Launch workers"
dask worker \
	--nworkers ${DASK_WORKERS} \
	--local-directory /tmp \
	--scheduler-file=${SCHEFILE} &
dask_worker_pid=$!

sleep 10

echo "Launch analytics"
python3 python/analytics.py apps/gys_io.yaml &
analytics_pid=$!

echo "Launch simu"
mpirun -n $SIMU_NODES build/apps/gys_io apps/gys_io.yaml apps/pdi_default.yaml & 
simu_pid=$!

wait ${analytics_pid}
echo "Analytics over"
wait ${simu_pid}
echo "Simulation over"

kill -9 ${dask_worker_pid} ${dask_sch_pid}
