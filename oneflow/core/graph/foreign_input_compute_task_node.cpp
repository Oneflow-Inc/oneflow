#include "oneflow/core/graph/foreign_input_compute_task_node.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kForeignInput, 1);
}
