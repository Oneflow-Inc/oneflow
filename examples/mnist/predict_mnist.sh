set -e

rm -rf ./predict_log ./core.* 

mkdir predict_log

GLOG_logtostderr=0 GLOG_log_dir=./predict_log GLOG_logbuflevel=-1 GLOG_v=0 ./scheduler \
    -job_conf="./predict_job.prototxt" \
    -this_machine_name="centos-0"
