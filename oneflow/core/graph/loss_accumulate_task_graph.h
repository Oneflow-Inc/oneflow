#ifndef ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class LossAccTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccTaskGraph);
  LossAccTaskGraph() = delete;
  ~LossAccTaskGraph() = default;

  LossAccTaskGraph(const std::string& name, CompTaskNode* loss_task);

  const char* TypeName() const override { return "LossAccTaskGraph"; }
  CompTaskNode* loss_task() { return loss_task_; }

 private:
  void BuildTaskGraph();

  CompTaskNode* loss_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_TASK_GRAPH_H_
