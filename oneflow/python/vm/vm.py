from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util

@oneflow_export('vm.run_instruction_list')
def vm_run(instruction_list):
    c_api_util.RunVmInstructionList(instruction_list)
