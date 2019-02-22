#!/bin/bash

#0 prepare the host list for training
#comment unused hosts with `#`
declare -a host_list=(
                  "192.168.1.11"
                  "192.168.1.12"
                  "192.168.1.13"
                  "192.168.1.14"
                  "192.168.1.15"
                  "192.168.1.16"
                  )
echo "Working on hosts:${host_list[@]}"

#1 prepare workspace_path folder on each host
workspace_path=~/oneflow_workdir

for host in "${host_list[@]}"
do
  ssh ${USER}@${host} "mkdir -p ${workspace_path}"
done

#2 copy files to each host and start work
if [ -n "$1" ]
then
  exec_path=$1
else
  exec_path=~/oneflow/build/bin/oneflow
fi

for host in "${host_list[@]}"
do
  ssh ${USER}@${host} "rm -rf ${workspace_path}/*"
  echo "copy files to ${host}"
  scp ${exec_path} ./*.prototxt ./train.sh ${USER}@${host}:${workspace_path}
  echo "start training on ${host}"
  ssh ${USER}@${host} "(cd ${workspace_path}; nohup ./train.sh --nocopy &) >/dev/null"
done
