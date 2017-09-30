set -e

rm -rf ./predict_log ./core.* ./predict_result/

GLOG_logtostderr=0 GLOG_log_dir=./predict_log GLOG_logbuflevel=-1 GLOG_v=0 ./scheduler \
    -job_conf_filepath="./predict_job.prototxt" \
    -this_machine_name="centos-0"
