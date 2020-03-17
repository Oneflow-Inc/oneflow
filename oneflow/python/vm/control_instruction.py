from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.vm.vm_instruction_pb2 as vm_instruction_util
import oneflow.python.vm.vm_instruction_build_context as vm_instr_build_ctx
import oneflow as flow

@oneflow_export("vm.new_host_symbol")
def new_host_symbol(symbol):
    vm_instr_proto = vm_instruction_util.VmInstructionProto()
    vm_instr_proto.vm_instr_type_name = "NewSymbol"
    symbol_operand = vm_instruction_util.VmInstructionOperandProto()
    symbol_operand.uint64_i_operand = symbol
    parallel_id_operand = vm_instruction_util.VmInstructionOperandProto()
    parallel_id_operand.int64_i_operand = 1
    vm_instr_proto.operand.extend([symbol_operand, parallel_id_operand])
    vm_instr_build_ctx.vm_instruction_list.vm_instruction.append(vm_instr_proto)

@oneflow_export("vm.delete_host_symbol")
def delete_host_symbol(symbol):
    vm_instr_proto = vm_instruction_util.VmInstructionProto()
    vm_instr_proto.vm_instr_type_name = "DeleteSymbol"
    symbol_operand = vm_instruction_util.VmInstructionOperandProto()
    symbol_operand.mutable_operand.logical_object_id = symbol
    symbol_operand.mutable_operand.all_parallel_id.SetInParent()
    vm_instr_proto.operand.append(symbol_operand)
    vm_instr_build_ctx.vm_instruction_list.vm_instruction.append(vm_instr_proto)

@oneflow_export("vm.new_device_symbol")
def new_device_symbol(symbol):
  TODO()

@oneflow_export("vm.new_local_host_symbol")
def new_local_host_symbol(symbol):
  TODO()

@oneflow_export("vm.new_local_device_symbol")
def new_local_device_symbol(symbol):
  TODO()
