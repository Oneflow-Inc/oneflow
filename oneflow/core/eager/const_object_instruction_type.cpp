#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/const_object_instruction_type.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

COMMAND(Global<vm::Storage<JobConfigProto>>::SetAllocated(new vm::Storage<JobConfigProto>()));
using JobDescInstr = vm::InitConstObjectInstructionType<JobDesc, JobConfigProto>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescSymbol"));
COMMAND(vm::RegisterLocalInstructionType<JobDescInstr>("LocalInitJobDescSymbol"));

COMMAND(Global<vm::Storage<OperatorConf>>::SetAllocated(new vm::Storage<OperatorConf>()));
using OperatorConfInstr = vm::InitConstObjectInstructionType<OperatorConf, OperatorConf>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfSymbol"));
COMMAND(vm::RegisterLocalInstructionType<OperatorConfInstr>("LocalInitOperatorConfSymbol"));

}  // namespace eager
}  // namespace oneflow
