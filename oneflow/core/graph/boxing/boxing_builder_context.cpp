#include <utility>

#include "oneflow/core/graph/boxing/boxing_builder_context.h"

namespace oneflow {

BasicBoxingBuilderCtx::BasicBoxingBuilderCtx(
    TaskGraph* task_graph, std::function<int64_t(const TaskNode*)> func_allocate_cpu_thrd_id)
    : task_graph_(task_graph), func_allocate_cpu_thrd_id_(std::move(func_allocate_cpu_thrd_id)) {}

TaskGraph* BasicBoxingBuilderCtx::task_graph() { return task_graph_; }

int64_t BasicBoxingBuilderCtx::AllocateCpuThrdId(const oneflow::TaskNode* task_node) {
  return func_allocate_cpu_thrd_id_(task_node);
}

}  // namespace oneflow
