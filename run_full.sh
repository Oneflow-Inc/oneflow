#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 115 2500mb 40 tmp --no-dataloader --no-o-one
CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 115 2500mb 40 tmp --no-dataloader --no-o-one --no-lr
CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 115 2500mb 40 tmp --no-dataloader --no-o-one --no-fbip
CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 115 2500mb 40 tmp --no-dataloader --no-o-one --no-fbip --no-lr
CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 115 2500mb 40 tmp --no-dataloader --no-o-one --no-fbip --no-lr --no-allo

# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 4000mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 3500mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv --no-lr
# python3 $SCRIPT_DIR/run.py 1 115 4000mb 40 --no-o-one --no-allocator
