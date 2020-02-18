#! /bin/sh
#
# build_user_op.sh
# Copyright (C) 2019 guoran <guoran@oneflow-15>
#
# Distributed under terms of the MIT license.
#


OF_CFLAGS=( $(python -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_compile_flags()))') )
OF_LFLAGS=( $(python -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_link_flags()))') )

g++ -std=c++11 -shared predict_decoder_op.cpp -o predict_decoder_op.so -fPIC ${OF_CFLAGS[@]} ${OF_LFLAGS[@]} -I/home/guoran/git-repo/darknet/include -L/home/guoran/git-repo/darknet/ -l:libdarknet.so -O2
