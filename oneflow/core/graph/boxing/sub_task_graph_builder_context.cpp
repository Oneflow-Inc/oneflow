#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

SubTskGphBuilderCtx::SubTskGphBuilderCtx(TaskGraph* task_graph) : task_graph_(task_graph) {}

TaskGraph* SubTskGphBuilderCtx::task_graph() { return task_graph_; }

}  // namespace oneflow
