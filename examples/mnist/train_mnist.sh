set -e

rm -rf ./train_log ./core.* ./snapshots

mkdir train_log

GLOG_logtostderr=0 GLOG_log_dir=./train_log GLOG_logbuflevel=-1 GLOG_v=0 ./scheduler \
    -job_conf="./train_job.prototxt" \
    -this_machine_name="centos-0"
