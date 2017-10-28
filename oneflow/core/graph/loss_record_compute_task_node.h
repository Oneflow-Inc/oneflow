#ifndef ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class LossRecordCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordCompTaskNode);
  LossRecordCompTaskNode() = default;
  ~LossRecordCompTaskNode() = default;

  void NewAllProducedRegst() override;
  TodoTaskType GetTaskType() const override {
    return TodoTaskType::kLossRecord;
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMPUTE_TASK_NODE_H_
