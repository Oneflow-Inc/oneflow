#ifndef ONEFLOW_CORE_GRAPH_NCCL_ALLREDUCE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_ALLREDUCE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NcclAllReduceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllReduceCompTaskNode);
  NcclAllReduceCompTaskNode() = default;
  ~NcclAllReduceCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kNcclAllReduce; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_ALLREDUCE_COMPUTE_TASK_NODE_H_
