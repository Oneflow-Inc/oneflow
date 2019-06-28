#include "oneflow/core/graph/model_save_compute_task_node.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kMdSave, ([]() -> size_t {
                                  return Global<ResourceDesc>::Get()->MaxMdSaveWorkerNum();
                                }));
}
