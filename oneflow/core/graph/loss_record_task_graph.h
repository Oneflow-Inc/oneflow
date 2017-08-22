#ifndef ONEFLOW_CORE_LOSS_RECORD_TASK_GRAPH_H_
#define ONEFLOW_CORE_LOSS_RECORD_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class LossRecordTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordTaskGraph);
  LossRecordTaskGraph() = delete;
  ~LossRecordTaskGraph() = default;

  LossRecordTaskGraph(const std::string& name,
                      const std::vector<TaskNode*>& sorted_loss_acc_task);

  const char* TypeName() const override { return "LossRecordTaskGraph"; }

  CompTaskNode* GetLossAccCompTaskNodeFromParallelId(int64_t parallel_id) {
    return sorted_loss_acc_tasks_.at(parallel_id);
  }

 private:
  void BuildTaskGraph(const std::vector<TaskNode*>& sorted_loss_acc_task);
  std::vector<CompTaskNode*> sorted_loss_acc_tasks_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LOSS_RECORD_TASK_GRAPH_H_
