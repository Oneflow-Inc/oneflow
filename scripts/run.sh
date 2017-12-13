set -e
set -x

declare -a hosts=("192.168.1.11")

ONEFLOW_CMD='nohup ./oneflow -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 -job_conf_filepath=./job.prototxt'

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
  ssh $USER@$host "cd ~/oneflow_temp; $ONEFLOW_CMD -this_machine_name=$host 1>./stdout 2>&1 </dev/null &"
done
