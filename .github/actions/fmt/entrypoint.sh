#!/bin/sh -l

set -ex
python3 ci/check/run_license_format.py -i oneflow -c
python3 ci/check/run_clang_format.py --clang_format_binary clang-format --source_dir oneflow
python3 ci/check/run_py_format.py --source_dir $PWD
time=$(date)
echo "::set-output name=time::$time"
