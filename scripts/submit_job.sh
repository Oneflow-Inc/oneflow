if [ "$1" == "--help" ]; then
  echo "Usage: `basename $0` /path/to/job.prototxt"
  exit 0
fi

job_filepath=$1

GetFilePathByName() {
  grep $1 $job_filepath | cut -d " " -f 2 | cut -d "\"" -f 2
}

dlnet_filepath=$(GetFilePathByName dlnet_filepath)
resource_filepath=$(GetFilePathByName resource_filepath)
placement_filepath=$(GetFilePathByName placement_filepath)

machine_addr_list=($(grep addr $resource_filepath | cut -d "\"" -f 2))
machine_name_list=($(grep name $resource_filepath | cut -d "\"" -f 2))

for machine_addr in "${machine_addr_list[@]}"
do
  ssh $USER@$machine_addr "mkdir ~/oneflow_temp"
done

oneflow_cmd="nohup ./oneflow -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 -job_conf_filepath=$job_filepath"

for ((machine_idx=0; machine_idx<${#machine_addr_list[@]}; ++machine_idx));
do
  ssh $USER@${machine_addr_list[$machine_idx]} 'rm -rf ~/oneflow_temp/*'
  scp ./oneflow $job_filepath $dlnet_filepath $resource_filepath $placement_filepath $USER@${machine_addr_list[$machine_idx]}:~/oneflow_temp
  ssh $USER@${machine_addr_list[$machine_idx]} "cd ~/oneflow_temp; $oneflow_cmd -this_machine_name=${machine_name_list[$machine_idx]} 1>/dev/null 2>&1 </dev/null &"
done
