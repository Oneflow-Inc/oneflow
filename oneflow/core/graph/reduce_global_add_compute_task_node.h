#ifndef ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceGlobalAddCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAddCompTaskNode);
  ReduceGlobalAddCompTaskNode() = default;
  ~ReduceGlobalAddCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceGlobalAdd; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;

  PbRf<int64_t> in_parallel_ids_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_
