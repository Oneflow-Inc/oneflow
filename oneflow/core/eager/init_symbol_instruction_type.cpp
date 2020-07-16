#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"

namespace oneflow {
namespace eager {

COMMAND(Global<vm::SymbolStorage<Scope>>::SetAllocated(new vm::SymbolStorage<Scope>()));
using ScopeInstr = vm::InitSymbolInstructionType<Scope>;
COMMAND(vm::RegisterInstructionType<ScopeInstr>("InitScopeSymbol"));

COMMAND(Global<vm::SymbolStorage<JobDesc>>::SetAllocated(new vm::SymbolStorage<JobDesc>()));
using JobDescInstr = vm::InitSymbolInstructionType<JobDesc>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescSymbol"));

COMMAND(
    Global<vm::SymbolStorage<OperatorConf>>::SetAllocated(new vm::SymbolStorage<OperatorConf>()));
using OperatorConfInstr = vm::InitSymbolInstructionType<OperatorConf>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfSymbol"));

COMMAND(Global<vm::SymbolStorage<OpParallelAttribute>>::SetAllocated(
    new vm::SymbolStorage<OpParallelAttribute>()));
using OpParallelAttributeInstr = vm::InitSymbolInstructionType<OpParallelAttribute>;
COMMAND(vm::RegisterInstructionType<OpParallelAttributeInstr>("InitOpParallelAttributeSymbol"));

}  // namespace eager
}  // namespace oneflow
