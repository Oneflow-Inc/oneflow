#ifndef ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class DistributeSplitCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitCompTaskNode);
  DistributeSplitCompTaskNode() = default;
  ~DistributeSplitCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  bool IsReadyForBuild() override;

  TaskType GetTaskType() const override { return TaskType::kDistributeSplit; }
  bool HasBackwardCompTaskNode();

 private:
  void BuildExecGphAndRegst() override;
  void BuildExecGphStructAndBindInRegst();
  void BuildOutRegst();
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_
