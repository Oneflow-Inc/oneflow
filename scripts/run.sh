set -e
set -x

declare -a hosts=("192.168.1.11")

ONEFLOW_CMD='GLOG_logtostderr=0 GLOG_log_dir=./log GLOG_v=0 GLOG_logbuflevel=-1 nohup ./oneflow -job_conf_filepath=./job.prototxt'

set +e
for host in "${hosts[@]}"
do
  ssh $USER@$host "mkdir ~/oneflow_temp"
done
set -e

for host in "${hosts[@]}"
do
  ssh $USER@$host 'rm -rf ~/oneflow_temp/*'
  scp ./oneflow ./*.prototxt $USER@$host:~/oneflow_temp
  ssh $USER@$host "cd ~/oneflow_temp; $ONEFLOW_CMD -this_machine_name=$host 1>/dev/null 2>&1 </dev/null &"
done
