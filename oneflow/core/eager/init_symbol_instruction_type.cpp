#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

COMMAND(Global<vm::Storage<JobConfigProto>>::SetAllocated(new vm::Storage<JobConfigProto>()));
using JobDescInstr = vm::InitSymbolInstructionType<JobDesc, JobConfigProto>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescSymbol"));

COMMAND(Global<vm::Storage<OperatorConf>>::SetAllocated(new vm::Storage<OperatorConf>()));
using OperatorConfInstr = vm::InitSymbolInstructionType<OperatorConf, OperatorConf>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfSymbol"));

}  // namespace eager
}  // namespace oneflow
