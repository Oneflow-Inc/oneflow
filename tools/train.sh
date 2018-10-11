#! /bin/sh

/usr/sbin/fuser -k 6789/tcp

if [ "$1" != "--nocopy" ]
then
  if [ -n "$1" ]
  then
    exec_path=$1
  else
    exec_path=~/oneflow/build/bin/oneflow
  fi
  cp ${exec_path} .
fi

rm -rf ./log/${HOSTNAME} ./core.*

START=$(date +%s)

GLOG_logtostderr=0 GLOG_log_dir=./log GLOG_logbuflevel=-1 GLOG_v=0 \
    ./oneflow -job_conf="./job.prototxt"

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Done in ${DIFF} seconds"
dir=${PWD##*/}
out=($(wc ./log/$HOSTNAME/oneflow.INFO))
cat >> record.txt <<EOF
${dir}, $(date), Training time: ${DIFF}, number of lines of log file: ${out[0]}
EOF
