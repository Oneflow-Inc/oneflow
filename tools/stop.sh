#!/bin/bash

cmd=oneflow
pids=`ps -u ${USER} | grep ${cmd} | awk '{print $1}'`
for pid in ${pids}
do
  kill -9 ${pid}
  echo "Kill ${cmd} process PID: ${pid}"
  sleep 1
done
