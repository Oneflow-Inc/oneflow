set -e

rm -rf ./train_log ./core.* ./snapshots

mkdir train_log

GLOG_logtostderr=0 GLOG_log_dir=./train_log ./compiler \
    -job_conf_filepath="./train_job.prototxt" \
    -plan_filepath="./train_plan" \

GLOG_logtostderr=0 GLOG_log_dir=./train_log GLOG_logbuflevel=-1 GLOG_v=0 ./runtime \
    -plan_filepath="./train_plan" \
    -this_machine_name="centos-0"
