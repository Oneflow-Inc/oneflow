if [[ $1 == 'enable' ]]; then
    oneflow_dir=$(python3 -c 'import oneflow; print(oneflow.__path__[0])')
    oneflow_tmp=$oneflow_dir/oneflow_tmp
    torch_dir=$oneflow_tmp/torch
    if [[ ! -d "$torch_dir" ]]; then mkdir -p $torch_dir; fi
    echo 'from oneflow import *' >$torch_dir/__init__.py
    export PYTHONPATH=$oneflow_tmp:$PYTHONPATH
    echo enabling import torch as flow
elif [[ $1 == 'disable' ]]; then
    export PYTHONPATH=$(p=$(echo $PYTHONPATH | tr ":" "\n" | grep -v "oneflow_tmp" | tr "\n" ":"); echo ${p%:})
    echo disabling import torch as flow
fi
