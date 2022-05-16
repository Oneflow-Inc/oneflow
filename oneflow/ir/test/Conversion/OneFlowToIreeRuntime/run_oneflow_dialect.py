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


compiled_flatbuffer = compile_str(SIMPLE_RELU_TOSA_DIALECT, target_backends=['dylib-llvm-aot'], input_type='tosa')
vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)

config = ireert.Config('dylib')
ctx = ireert.SystemContext(config=config)

ctx.add_vm_module(vm_module)
print("INVOKE simple_relu")

tensor = flow.tensor([[-1., -1.], [0., 1.]]).cuda()

input = tensor.cpu().detach().numpy()
f = ctx.modules.module['main']
output = f(input).to_host()

print("Results:", output)