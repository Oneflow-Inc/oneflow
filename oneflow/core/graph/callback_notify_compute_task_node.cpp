#include "oneflow/core/graph/callback_notify_compute_task_node.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kCallbackNotify, 1);
}
