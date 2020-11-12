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

namespace oneflow {

namespace {

void SetSoleMirroredOperand(vm::cfg::OperandProto* operand, int64_t symbol_id) {
  operand->set_logical_object_id(symbol_id);
  operand->mutable_sole_mirrored_object();
}

void InitSymbolOperand(vm::cfg::InstructionOperandProto* instr_operand, int64_t symbol_id) {
  SetSoleMirroredOperand(instr_operand->mutable_init_symbol_operand(), symbol_id);
}

}  // namespace

Maybe<int64_t> InstructionsBuilder::CreateScopeSymbolId(const cfg::ScopeProto& scope_proto) {
  int64_t symbol_id = JUST(mut_id_generator()->NewSymbolId());
  {
    auto* instruction = mut_instruction_list()->mutable_instruction()->Add();
    instruction->set_instr_type_name("InitScopeSymbol");
    InitSymbolOperand(instruction->mutable_operand()->Add(), symbol_id);
  }
  {
    auto* eager_symbol = mut_eager_symbol_list()->mutable_eager_symbol()->Add();
    eager_symbol->set_symbol_id(symbol_id);
    eager_symbol->mutable_scope_symbol()->CopyFrom(scope_proto);
  }
  return symbol_id;
}

}  // namespace oneflow
