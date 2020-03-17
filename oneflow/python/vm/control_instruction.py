from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.vm.instruction_pb2 as instruction_util
import oneflow.python.vm.instruction_build_context as instr_build_ctx
import oneflow as flow

@oneflow_export("vm.new_host_symbol")
def new_host_symbol(symbol):
    instr_proto = instruction_util.InstructionProto()
    instr_proto.instr_type_name = "NewSymbol"
    symbol_operand = instruction_util.InstructionOperandProto()
    symbol_operand.uint64_i_operand = symbol
    parallel_id_operand = instruction_util.InstructionOperandProto()
    parallel_id_operand.int64_i_operand = 1
    instr_proto.operand.extend([symbol_operand, parallel_id_operand])
    instr_build_ctx.instruction_list.instruction.append(instr_proto)

@oneflow_export("vm.delete_host_symbol")
def delete_host_symbol(symbol):
    instr_proto = instruction_util.InstructionProto()
    instr_proto.instr_type_name = "DeleteSymbol"
    symbol_operand = instruction_util.InstructionOperandProto()
    symbol_operand.mutable_operand.logical_object_id = symbol
    symbol_operand.mutable_operand.all_parallel_id.SetInParent()
    instr_proto.operand.append(symbol_operand)
    instr_build_ctx.instruction_list.instruction.append(instr_proto)

@oneflow_export("vm.new_device_symbol")
def new_device_symbol(symbol):
  TODO()

@oneflow_export("vm.new_local_host_symbol")
def new_local_host_symbol(symbol):
  TODO()

@oneflow_export("vm.new_local_device_symbol")
def new_local_device_symbol(symbol):
  TODO()
