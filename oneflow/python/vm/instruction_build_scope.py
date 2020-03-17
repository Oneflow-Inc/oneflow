from __future__ import absolute_import

from contextlib import contextmanager
from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.vm.instruction_pb2 as instruction_util
import oneflow.python.vm.instruction_build_context as instr_build_ctx
import oneflow.python.framework.session_context as session_ctx

@oneflow_export("vm.new_instruction_list")
def new_instruction_list():
    return instruction_util.InstructionListProto()

@oneflow_export("vm.instruction_build_scope")
@contextmanager
def instruction_build_scope(instruction_list):
    assert session_ctx.GetDefaultSession().is_running, "no session running"
    assert instr_build_ctx.instruction_list is None,\
        "no nested vm instruction build scope supported"
    instr_build_ctx.instruction_list = instruction_list
    yield
    instr_build_ctx.instruction_list = None
