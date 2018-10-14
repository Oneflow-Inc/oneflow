#ifndef ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class RepeatBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatBackwardCompTaskNode);
  RepeatBackwardCompTaskNode() = default;
  ~RepeatBackwardCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kRepeatBackward; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kCompute; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_COMPUTE_TASK_NODE_H_
