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

}  // namespace oneflow
