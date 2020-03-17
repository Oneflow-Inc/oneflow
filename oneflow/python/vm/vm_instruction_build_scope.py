from __future__ import absolute_import

from contextlib import contextmanager
from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.vm.vm_instruction_pb2 as vm_instruction_util
import oneflow.python.vm.vm_instruction_build_context as vm_instr_build_ctx
import oneflow.python.framework.session_context as session_ctx

@oneflow_export("vm.new_instruction_list")
def new_vm_instruction_list():
    return vm_instruction_util.VmInstructionListProto()

@oneflow_export("vm.instruction_build_scope")
@contextmanager
def vm_instruction_build_scope(vm_instruction_list):
    assert session_ctx.GetDefaultSession().is_running, "no session running"
    assert vm_instr_build_ctx.vm_instruction_list is None,\
        "no nested vm instruction build scope supported"
    vm_instr_build_ctx.vm_instruction_list = vm_instruction_list
    yield
    vm_instr_build_ctx.vm_instruction_list = None
