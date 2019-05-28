#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class SubTskGphBuilderCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilderCtx);
  SubTskGphBuilderCtx() = default;
  virtual ~SubTskGphBuilderCtx() = default;

  virtual TaskGraph* task_graph() = 0;
  virtual int64_t AllocateCpuThrdId(const TaskNode* task_node) = 0;
};

class BasicSubTskGphBuilderCtx final : public SubTskGphBuilderCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicSubTskGphBuilderCtx);
  BasicSubTskGphBuilderCtx(TaskGraph* task_graph,
                           std::function<int64_t(const TaskNode*)> func_allocate_cpu_thrd_id);
  ~BasicSubTskGphBuilderCtx() override = default;

 private:
  TaskGraph* task_graph() override;
  int64_t AllocateCpuThrdId(const TaskNode* task_node) override;

  TaskGraph* task_graph_;
  std::function<int64_t(const TaskNode*)> func_allocate_cpu_thrd_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
