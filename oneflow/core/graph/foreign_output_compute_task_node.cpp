#include "oneflow/core/graph/foreign_output_compute_task_node.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kForeignOutput, 1);
}
