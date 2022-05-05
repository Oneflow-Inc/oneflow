import os
import oneflow as flow

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"

class RELU(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class GraphToRun(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.fw = RELU()

    def build(self, x):
        return self.fw(x)

graph_to_run = GraphToRun()
x = flow.tensor([-1.], dtype=flow.float32)
y_lazy = graph_to_run(x)


from iree import runtime as ireert
from iree.compiler import compile_str

SIMPLE_RELU_ONEFLOW_DIALECT = '?'

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

import numpy as np
arg0 = np.array([[-1., -1.], [0., 1.]], dtype=np.float32)

f = ctx.modules.module['main']
results = f(arg0)
print("Results:", results)
