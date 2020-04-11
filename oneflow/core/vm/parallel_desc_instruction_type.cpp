#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/const_object_instruction_type.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

COMMAND(Global<Storage<ParallelConf>>::SetAllocated(new Storage<ParallelConf>()));
using ParallelDescInstr = InitConstObjectInstructionType<ParallelDesc, ParallelConf>;
COMMAND(RegisterInstructionType<ParallelDescInstr>("InitParallelDescObject"));
COMMAND(RegisterLocalInstructionType<ParallelDescInstr>("LocalInitParallelDescObject"));

}  // namespace vm
}  // namespace oneflow
