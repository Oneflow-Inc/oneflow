#!/bin/bash
#

##############################################
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

##############################################
#1 prepare oneflow_temp folder on each host
for host in "${host_list[@]}"
do
  ssh ${USER}@${host} "mkdir -p ~/oneflow_workdir"
done

##############################################
#2 copy files to each host and start work
if [ -n "$1" ]
then
  exec_path=$1
else
  exec_path=~/oneflow/build/bin/oneflow
fi

for host in "${host_list[@]}"
do
  ssh ${USER}@${host} "rm -rf ~/oneflow_workdir/*"
  echo "copy files to ${host}"
  scp ${exec_path} ./*.prototxt ./train.sh ${USER}@${host}:~/oneflow_workdir
  echo "start training on ${host}"
  ssh ${USER}@${host} "(cd ~/oneflow_workdir; nohup ./train.sh --nocopy &) >/dev/null"
done
