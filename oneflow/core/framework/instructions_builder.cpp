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
#include "oneflow/core/framework/symbol_storage_util.h"
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

void SetMirroredOperand(vm::cfg::OperandProto* operand, int64_t object_id) {
  operand->set_logical_object_id(object_id);
  operand->mutable_current_global_device_id();
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

Maybe<void> AddStringSymbol(int64_t symbol_id, const std::string& data) {
  JUST(Global<symbol::Storage<StringSymbol>>::Get()->Add(symbol_id, data));
  auto* id_cache = JUST(GlobalMaybe<symbol::IdCache<std::string>>());
  CHECK_OR_RETURN(!id_cache->Has(data));
  JUST(id_cache->FindOrCreate(data, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
  return Maybe<void>::Ok();
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

Maybe<int64_t> InstructionsBuilder::NewSymbolId() {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewObjectId(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  int64_t object_id = CHECK_JUST(id_generator_->NewObjectId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewObject");
  instruction.set_parallel_desc_symbol_id(CHECK_JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(object_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return object_id;
}

Maybe<StringSymbol> InstructionsBuilder::GetSymbol4String(std::string str) {
  if (JUST(HasSymbol<std::string>(str))) { return GetSymbol<std::string, StringSymbol>(str); }
  int64_t symbol_id = JUST(NewSymbolId4String(str));
  JUST(AddStringSymbol(symbol_id, str));
  return GetSymbol<std::string, StringSymbol>(str);
}

Maybe<JobDesc> InstructionsBuilder::GetJobConfSymbol(
    const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  if (JUST(HasSymbol<cfg::JobConfigProto>(*job_conf))) {
    return GetSymbol<cfg::JobConfigProto, JobDesc>(*job_conf);
  }
  int64_t symbol_id = JUST(NewSymbolId4JobConf(job_conf));
  JUST(AddSymbol<cfg::JobConfigProto, JobConfigProto, JobDesc>(symbol_id, *job_conf));
  return GetSymbol<cfg::JobConfigProto, JobDesc>(*job_conf);
}

Maybe<ParallelDesc> InstructionsBuilder::GetParallelDescSymbol(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  if (JUST(HasSymbol<cfg::ParallelConf>(*parallel_conf))) {
    return GetSymbol<cfg::ParallelConf, ParallelDesc>(*parallel_conf);
  }
  int64_t symbol_id = JUST(NewSymbolId4ParallelConf(parallel_conf));
  JUST(AddSymbol<cfg::ParallelConf, ParallelConf, ParallelDesc>(symbol_id, *parallel_conf));
  return GetSymbol<cfg::ParallelConf, ParallelDesc>(*parallel_conf);
}

Maybe<Scope> InstructionsBuilder::GetScopeSymbol(
    const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  if (JUST(HasSymbol<cfg::ScopeProto>(*scope_proto))) {
    return GetSymbol<cfg::ScopeProto, Scope>(*scope_proto);
  }
  int64_t symbol_id = JUST(NewSymbolId4Scope(scope_proto));
  JUST(AddSymbol<cfg::ScopeProto, ScopeProto, Scope>(symbol_id, *scope_proto));
  return GetSymbol<cfg::ScopeProto, Scope>(*scope_proto);
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4String(std::string str) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitStringSymbol(symbol_id, str));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4JobConf(
    const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitJobConfSymbol(symbol_id, job_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4ParallelConf(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  int64_t symbol_id = CHECK_JUST(id_generator_->NewSymbolId());
  JUST(NewParallelConfSymbol(symbol_id, parallel_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4Scope(
    const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(NewScopeSymbol(symbol_id, scope_proto));
  return symbol_id;
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::NewBlobObject(
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  int64_t object_id = JUST(NewObjectId(op_arg_parallel_attr->parallel_desc_symbol()));
  std::shared_ptr<compatible_py::BlobObject> obj = std::make_shared<compatible_py::BlobObject>(
      object_id, op_arg_parallel_attr, op_arg_blob_attr);
  obj->add_releaser(release_object_);
  return obj;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4OpNodeSignature(
    const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitOpNodeSignatureDescSymbol(symbol_id, op_node_signature_sym));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSharedOpKernelObjectId4ParallelConfSymbolId(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return NewObjectId(parallel_desc_sym);
}

Maybe<void> InstructionsBuilder::InitStringSymbol(int64_t symbol_id, std::string str) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitStringSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.set_string_symbol(str);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InitJobConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitJobDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_job_conf_symbol()->CopyFrom(*job_conf);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NewParallelConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewParallelDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_parallel_conf_symbol()->CopyFrom(*parallel_conf);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NewScopeSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitScopeSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_scope_symbol()->CopyFrom(*scope_proto);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InitOpNodeSignatureDescSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitOpNodeSignatureDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_op_node_signature_symbol()->CopyFrom(*op_node_signature_sym);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

std::shared_ptr<vm::cfg::InstructionOperandProto> DelObjectOperand(int64_t object_id) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetAllMirroredOperand(operand->mutable_mut_operand(), object_id);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> MutOperand(int64_t object_id) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_mut_operand(), object_id);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Int64Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->set_int64_operand(val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> InitSymbolOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetSoleMirroredOperand(operand->mutable_init_symbol_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> SymbolOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetSoleMirroredOperand(operand->mutable_symbol_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> ConstOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_const_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> OperandSeparator() {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->mutable_separator();
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Uint64Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->set_uint64_operand(val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Mut2Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_mut2_operand(), val);
  return operand;
}

}  // namespace oneflow
