import numpy as np
import oneflow as flow

from iree import runtime as ireert
from iree.compiler import compile_str

SIMPLE_RELU_ONEFLOW_DIALECT = '''
module  {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
'''

import subprocess
run_args = {}
run_args["stdout"] = subprocess.PIPE
run_args["stderr"] = subprocess.PIPE
cmd = ['oneflow-opt', '-lower-oneflow-to-tosa']
process = subprocess.run(cmd, input=SIMPLE_RELU_ONEFLOW_DIALECT.encode(), **run_args)
if process.returncode != 0:
  print('oneflow-opt failed')
  exit(1)
SIMPLE_RELU_TOSA_DIALECT = process.stdout



'''
CUDA GPU	Target Architecture
Nvidia K80	sm_35
Nvidia P100	sm_60
Nvidia V100	sm_70
Nvidia A100	sm_80
'''
'''
iree-compile \
    --iree-mlir-to-vm-bytecode-module \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=<...> \
    --iree-hal-cuda-disable-loop-nounroll-wa \
    iree_input.mlir -o mobilenet-cuda.vmfb
'''

'''
iree/tools/iree-run-module \
    --driver=cuda \
    --module_file=mobilenet-cuda.vmfb \
    --entry_function=predict \
    --function_input="1x224x224x3xf32=0"
'''


config = ireert.Config('cuda')
ctx = ireert.SystemContext(config=config)
compiled_flatbuffer = compile_str(SIMPLE_RELU_TOSA_DIALECT, target_backends=['cuda'], input_type='tosa')
vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)
ctx.add_vm_module(vm_module)
print("INVOKE simple_relu on cuda")

import torch
input = torch.tensor([[-1., -1.], [0., 1.]]).cuda()

input = input.cpu().detach().numpy()

f = ctx.modules.module['main']
output = f(input).to_host()

print("Results:", output)