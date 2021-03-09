"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import numpy as np

flow.config.machine_num(1)
flow.config.gpu_device_num(1)


@flow.function()
def Foo(x=flow.FixedTensorDef((2, 5))):
    return x


Foo(np.ones((2, 5), dtype=np.float32))

vm_instr_list = flow.vm.new_instruction_list()
with flow.vm.instruction_build_scope(vm_instr_list):
    flow.vm.new_host_symbol(9527)
    flow.vm.delete_host_symbol(9527)

print(vm_instr_list)
flow.vm.run_instruction_list(vm_instr_list)
