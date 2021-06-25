#!/bin/bash
set -xe
export PYTHONUNBUFFERED=1
src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

unset val_stdout val_stderr
eval "$(python3 -c 'import oneflow; print(oneflow.__path__[0])' \
        2> >(readarray -t val_stderr; typeset -p val_stderr) \
         > >(readarray -t val_stdout; typeset -p val_stdout) )"
cd ${val_stdout}
cd ..

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=${PWD} \
    --timeout=1 \
    --verbose \
    --chunk=1 \
    --doctest
