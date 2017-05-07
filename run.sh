rm -rf ./log ./core.*
mkdir log
GLOG_logtostderr=1 GLOG_log_dir=./log ./compiler -job_conf_filepath=../prototxt/job_conf.prototxt

if [ "$1" = "--withdot" ]
then
  for dot_file in `ls ./log/*.dot`
  do
    echo "process ${dot_file}"
    dot -Tpng -O $dot_file
  done
fi
