set -e
set -x

declare -a hosts=("192.168.1.11" "192.168.1.13")

SCHEDULER_CMD='GLOG_logtostderr=0 GLOG_log_dir=./log GLOG_v=0 GLOG_logbuflevel=-1 nohup ./scheduler.sh -job_conf_filepath=./job.prototxt'

set +e
for host in "${hosts[@]}"
do
  ssh $USER@$host "/usr/sbin/fuser -k 6666/tcp"
  ssh $USER@$host "mkdir ~/oneflow_temp"
done
set -e

for host in "${hosts[@]}"
do
  ssh $USER@$host 'rm -rf ~/oneflow_temp/*'
  scp ./scheduler.sh ./*.prototxt $USER@$host:~/oneflow_temp
  ssh $USER@$host "cd ~/oneflow_temp; $SCHEDULER_CMD -this_machine_name=$host 1>scheduler.sh.log 2>&1 </dev/null &"
done
