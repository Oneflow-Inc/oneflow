#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace pthread_fork {

static bool is_fork = false;

bool IsFork() { return is_fork; }
static void SetIsFork() { is_fork = true; }

void RegisterForkCallback() { pthread_atfork(nullptr, nullptr, SetIsFork); }
COMMAND(RegisterForkCallback());

}  // namespace pthread_fork

}  // namespace oneflow
