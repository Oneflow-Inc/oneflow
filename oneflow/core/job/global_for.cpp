#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

COMMAND(Global<bool, EagerExecution<ForEnv>>::SetAllocated(new bool(false)));
// COMMAND(Global<bool, EagerExecution<ForSession>>::SetAllocated(new bool(false)));

}  // namespace oneflow
