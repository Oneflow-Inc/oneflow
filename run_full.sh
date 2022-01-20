#!/usr/bin/env bash

set -uex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one
python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-add-n --high-conv --nlr
python3 $SCRIPT_DIR/run.py 10 115 4000mb 40 --no-o-one --high-add-n --high-conv --nlr
python3 $SCRIPT_DIR/run.py 10 115 3500mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv --no-lr
# python3 $SCRIPT_DIR/run.py 1 115 4000mb 40 --no-o-one --no-allocator
