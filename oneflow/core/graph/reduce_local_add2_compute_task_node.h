#ifndef ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD2_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD2_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceLocalAdd2CompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceLocalAdd2CompTaskNode);
  ReduceLocalAdd2CompTaskNode() = default;
  ~ReduceLocalAdd2CompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceLocalAdd2; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD2_COMPUTE_TASK_NODE_H_
