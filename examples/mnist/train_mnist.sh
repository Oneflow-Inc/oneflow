set -e

rm -rf ./train_log ./core.* ./snapshots

GLOG_logtostderr=0 GLOG_log_dir=./train_log GLOG_v=0 GLOG_logbuflevel=-1 ./scheduler \
    -job_conf_filepath="./train_job.prototxt" \
    -this_machine_name="centos-0"
