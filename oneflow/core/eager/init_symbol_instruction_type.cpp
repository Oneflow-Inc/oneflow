#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

COMMAND(Global<vm::SymbolStorage<ScopeProto>>::SetAllocated(new vm::SymbolStorage<ScopeProto>()));
using ScopeInstr = vm::InitSymbolInstructionType<Scope, ScopeProto>;
COMMAND(vm::RegisterInstructionType<ScopeInstr>("InitScopeSymbol"));

COMMAND(Global<vm::SymbolStorage<JobConfigProto>>::SetAllocated(
    new vm::SymbolStorage<JobConfigProto>()));
using JobDescInstr = vm::InitSymbolInstructionType<JobDesc, JobConfigProto>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescSymbol"));

COMMAND(
    Global<vm::SymbolStorage<OperatorConf>>::SetAllocated(new vm::SymbolStorage<OperatorConf>()));
using OperatorConfInstr = vm::InitSymbolInstructionType<OperatorConf, OperatorConf>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfSymbol"));

}  // namespace eager
}  // namespace oneflow
