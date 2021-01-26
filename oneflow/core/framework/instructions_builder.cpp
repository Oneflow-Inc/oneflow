/*
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
*/
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"

namespace oneflow {

namespace {

void SetSoleMirroredOperand(vm::cfg::OperandProto* operand, int64_t symbol_id) {
  operand->set_logical_object_id(symbol_id);
  operand->mutable_sole_mirrored_object();
}

void SetInitSymbolOperand(vm::cfg::InstructionOperandProto* instr_operand, int64_t symbol_id) {
  SetSoleMirroredOperand(instr_operand->mutable_init_symbol_operand(), symbol_id);
}

void SetInt64Operand(vm::cfg::InstructionOperandProto* instr_operand, int64_t symbol_id) {
  instr_operand->set_int64_operand(symbol_id);
}

void SetAllMirroredOperand(vm::cfg::OperandProto* operand, int64_t object_id) {
  operand->set_logical_object_id(object_id);
  operand->mutable_all_mirrored_object();
}

vm::cfg::InstructionOperandProto DelObjectOperand(int64_t object_id) {
  vm::cfg::InstructionOperandProto operand;
  SetAllMirroredOperand(operand.mutable_mut_operand(), object_id);
  return operand;
}

void SetMirroredOperand(vm::cfg::OperandProto* operand, int64_t object_id) {
  operand->set_logical_object_id(object_id);
  operand->mutable_current_global_device_id();
}

vm::cfg::InstructionOperandProto MutOperand(int64_t object_id) {
  vm::cfg::InstructionOperandProto operand;
  SetMirroredOperand(operand.mutable_mut_operand(), object_id);
  return operand;
}

vm::cfg::InstructionOperandProto Int64Operand(int64_t val) {
  vm::cfg::InstructionOperandProto operand;
  operand.set_int64_operand(val);
  return operand;
}

vm::cfg::InstructionOperandProto InitSymbolOperand(int64_t val) {
  vm::cfg::InstructionOperandProto operand;
  SetSoleMirroredOperand(operand.mutable_init_symbol_operand(), val);
  return operand;
}

Maybe<int64_t> NewSymbolId(vm::IdGenerator* id_generator,
                           vm::cfg::InstructionListProto* instruction_list) {
  int64_t symbol_id = JUST(id_generator->NewSymbolId());
  auto* instruction = instruction_list->mutable_instruction()->Add();
  instruction->set_instr_type_name("NewSymbol");
  SetInt64Operand(instruction->mutable_operand()->Add(), symbol_id);
  return symbol_id;
}

template<typename T>
const char* GetInstrTypeName();

template<>
const char* GetInstrTypeName<cfg::JobConfigProto>() {
  return "InitJobDescSymbol";
}
template<>
const char* GetInstrTypeName<cfg::ParallelConf>() {
  return "NewParallelDescSymbol";
}
template<>
const char* GetInstrTypeName<cfg::ScopeProto>() {
  return "InitScopeSymbol";
}

template<typename T>
T* MutEagerSymbolConf(eager::cfg::EagerSymbol*);

template<>
cfg::JobConfigProto* MutEagerSymbolConf<cfg::JobConfigProto>(
    eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_job_conf_symbol();
}

template<>
cfg::ParallelConf* MutEagerSymbolConf<cfg::ParallelConf>(eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_parallel_conf_symbol();
}

template<>
cfg::ScopeProto* MutEagerSymbolConf<cfg::ScopeProto>(eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_scope_symbol();
}

}  // namespace

namespace detail {

template<typename T>
Maybe<int64_t> CreateSymbolIdHelper<T>::Call(vm::IdGenerator* id_generator,
                                             vm::cfg::InstructionListProto* instruction_list,
                                             eager::cfg::EagerSymbolList* eager_symbol_list,
                                             const T& conf) {
  int64_t symbol_id = JUST(NewSymbolId(id_generator, instruction_list));
  {
    auto* instruction = instruction_list->mutable_instruction()->Add();
    instruction->set_instr_type_name(GetInstrTypeName<T>());
    SetInitSymbolOperand(instruction->mutable_operand()->Add(), symbol_id);
  }
  {
    auto* eager_symbol = eager_symbol_list->mutable_eager_symbol()->Add();
    eager_symbol->set_symbol_id(symbol_id);
    MutEagerSymbolConf<T>(eager_symbol)->CopyFrom(conf);
  }
  return symbol_id;
}

template struct CreateSymbolIdHelper<cfg::JobConfigProto>;
template struct CreateSymbolIdHelper<cfg::ScopeProto>;

template<>
Maybe<int64_t> CreateSymbolIdHelper<cfg::ParallelConf>::Call(
    vm::IdGenerator* id_generator, vm::cfg::InstructionListProto* instruction_list,
    eager::cfg::EagerSymbolList* eager_symbol_list, const cfg::ParallelConf& conf) {
  int64_t symbol_id = JUST(id_generator->NewSymbolId());
  {
    auto* instruction = instruction_list->mutable_instruction()->Add();
    instruction->set_instr_type_name(GetInstrTypeName<cfg::ParallelConf>());
    SetInt64Operand(instruction->mutable_operand()->Add(), symbol_id);
  }
  {
    auto* eager_symbol = eager_symbol_list->mutable_eager_symbol()->Add();
    eager_symbol->set_symbol_id(symbol_id);
    MutEagerSymbolConf<cfg::ParallelConf>(eager_symbol)->CopyFrom(conf);
  }
  return symbol_id;
}

}  // namespace detail

int64_t InstructionsBuilder::NewSymbolId() {
  int64_t symbol_id = CHECK_JUST(id_generator_->NewSymbolId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(Int64Operand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return symbol_id;
}

int64_t InstructionsBuilder::NewObjectId(const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  int64_t object_id = CHECK_JUST(id_generator_->NewObjectId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewObject");
  instruction.set_parallel_desc_symbol_id(CHECK_JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(Int64Operand(object_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return object_id;
}

int64_t InstructionsBuilder::NewSymbolId4OpNodeSignature(
    std::shared_ptr<cfg::OpNodeSignature> op_node_signature_sym) {
  int64_t symbol_id = this->NewSymbolId();
  InitOpNodeSignatureDescSymbol(symbol_id, op_node_signature_sym);
  return symbol_id;
}

void InstructionsBuilder::InitOpNodeSignatureDescSymbol(
    int64_t symbol_id, std::shared_ptr<cfg::OpNodeSignature> op_node_signature_sym) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitOpNodeSignatureDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_op_node_signature_symbol()->CopyFrom(*op_node_signature_sym);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
}

void InstructionsBuilder::DeleteObject(compatible_py::BlobObject* blob_object) {
  _TryClearObject(blob_object);
  _DeleteObject(blob_object);
}

void InstructionsBuilder::_TryClearObject(compatible_py::BlobObject* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("TryClearObject");
  instruction.set_parallel_desc_symbol_id(
      CHECK_JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(MutOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
}

void InstructionsBuilder::_DeleteObject(compatible_py::BlobObject* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("DeleteObject");
  instruction.set_parallel_desc_symbol_id(
      CHECK_JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(DelObjectOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
}

}  // namespace oneflow
