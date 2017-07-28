set -e

rm -rf ./log ./log.tar.gz ./core.* ./snapshots

mkdir log
GLOG_logtostderr=0 GLOG_log_dir=./log ./compiler \
    -job_conf_filepath="../prototxt/single_machine_job.prototxt" \
    -plan_filepath="./plan" \

GLOG_logtostderr=0 GLOG_log_dir=./log GLOG_logbuflevel=-1 GLOG_v=4 ./runtime \
    -plan_filepath="./plan" \
    -this_machine_name="centos-0"

echo $?
