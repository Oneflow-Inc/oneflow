#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

class SubTskGphBuilderCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilderCtx);
  explicit SubTskGphBuilderCtx(TaskGraph* task_graph);
  ~SubTskGphBuilderCtx() = default;

  TaskGraph* task_graph();

 private:
  TaskGraph* task_graph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
