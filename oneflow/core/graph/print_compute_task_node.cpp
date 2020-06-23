#include "oneflow/core/graph/print_compute_task_node.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kPrint, []() -> size_t {
  return Global<ResourceDesc, ForSession>::Get()->MaxMdSaveWorkerNum();
});
}
