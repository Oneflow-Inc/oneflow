set -e

rm -rf ./predict_log ./core.* 

mkdir predict_log

GLOG_logtostderr=0 GLOG_log_dir=./predict_log ./compiler \
    -job_conf_filepath="./predict_job.prototxt" \
    -plan_filepath="./predict_plan" \

GLOG_logtostderr=0 GLOG_log_dir=./predict_log GLOG_logbuflevel=-1 GLOG_v=0 ./runtime \
    -plan_filepath="./predict_plan" \
    -this_machine_name="centos-0"
