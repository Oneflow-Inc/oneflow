import numpy as np

from iree import runtime as ireert
from iree.compiler import compile_str

SIMPLE_RELU_ASM = '''
module  {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
'''

compiled_flatbuffer = compile_str(SIMPLE_RELU_ASM, target_backends=['dylib-llvm-aot'], input_type='tosa')
vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)

config = ireert.Config('dylib')
ctx = ireert.SystemContext(config=config)

ctx.add_vm_module(vm_module)
print("INVOKE simple_relu")

arg0 = np.array([[-1., -1.], [0., 1.]], dtype=np.float32)

f = ctx.modules.module['main']
results = f(arg0)
print("Results:", results)
