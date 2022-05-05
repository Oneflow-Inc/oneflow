import numpy as np

from iree import runtime as ireert
from iree.compiler import compile_str

SIMPLE_RELU_ASM = '''
module {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 9223372036854775807 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
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
