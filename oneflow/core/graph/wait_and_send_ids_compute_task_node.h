#ifndef ONEFLOW_CORE_GRAPH_WAIT_AND_SEND_IDS_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_WAIT_AND_SEND_IDS_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class WaitAndSendIdsCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WaitAndSendIdsCompTaskNode);
  WaitAndSendIdsCompTaskNode() = default;
  ~WaitAndSendIdsCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  TaskType GetTaskType() const override { return TaskType::kWaitAndSendIds; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_WAIT_AND_SEND_IDS_COMPUTE_TASK_NODE_H_
