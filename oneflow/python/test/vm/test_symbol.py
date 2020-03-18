import oneflow as flow
import numpy as np

flow.config.machine_num(1)
flow.config.gpu_device_num(1)

@flow.function()
def Foo(x=flow.FixedTensorDef((2, 5))):
    return x

Foo(np.ones((2, 5), dtype=np.float32)); 

vm_instr_list = flow.vm.new_instruction_list()
with flow.vm.instruction_build_scope(vm_instr_list):
    flow.vm.new_host_symbol(9527)
    flow.vm.delete_host_symbol(9527)

print(vm_instr_list)
flow.vm.run_instruction_list(vm_instr_list)
