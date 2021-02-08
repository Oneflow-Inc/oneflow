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
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/op_conf_symbol.h"

namespace oneflow {
namespace eager {

COMMAND(Global<symbol::Storage<Scope>>::SetAllocated(new symbol::Storage<Scope>()));
using ScopeInstr = vm::InitSymbolInstructionType<Scope>;
COMMAND(vm::RegisterInstructionType<ScopeInstr>("InitScopeSymbol"));

COMMAND(Global<symbol::Storage<JobDesc>>::SetAllocated(new symbol::Storage<JobDesc>()));
using JobDescInstr = vm::InitSymbolInstructionType<JobDesc>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescSymbol"));

COMMAND(Global<symbol::Storage<OperatorConfSymbol>>::SetAllocated(
    new symbol::Storage<OperatorConfSymbol>()));

using OperatorConfInstr = vm::InitSymbolInstructionType<OperatorConfSymbol>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfSymbol"));

COMMAND(Global<symbol::Storage<OpNodeSignatureDesc>>::SetAllocated(
    new symbol::Storage<OpNodeSignatureDesc>()));
using OpNodeSignatureInstr = vm::InitSymbolInstructionType<OpNodeSignatureDesc>;
COMMAND(vm::RegisterInstructionType<OpNodeSignatureInstr>("InitOpNodeSignatureDescSymbol"));

}  // namespace eager
}  // namespace oneflow
