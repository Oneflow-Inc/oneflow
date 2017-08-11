rm -rf ./simple-log ./core.*
mkdir -p simple-log/dot/data && mkdir -p simple-log/dot/model && mkdir -p simple-log/dot/exec
GLOG_logtostderr=0 GLOG_log_dir=./simple-log ./compiler -job_conf_filepath=../prototxt/simple-job_conf.prototxt -plan_filepath=./simple-elf

if [ "$1" = "--withdot" ]
then
  for dot_file in `find ./simple-log -name *.dot`
  do
    echo "process ${dot_file}"
    dot -Tpng -O $dot_file
    dot -Tsvg -O $dot_file
  done
fi
