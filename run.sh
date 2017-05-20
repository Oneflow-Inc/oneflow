set -e

rm -rf ./log ./core.*
mkdir log && mkdir log/dot && mkdir log/dot/data && mkdir log/dot/model && mkdir "log/dot/exec"
GLOG_logtostderr=0 GLOG_log_dir=./log ./compiler -job_conf_filepath=../prototxt/multi_machine_job.prototxt -elf_filepath=./elf

exit 0

if [ "$1" = "--withdot" ]
then
  for dot_file in `find . -name *.dot`
  do
    echo "process ${dot_file}"
    dot -Tpng -O $dot_file
  done
fi

GLOG_logtostderr=1 GLOG_log_dir=./log ./elf_runner -elf_filepath=./elf -this_machine_name="centos-0"
