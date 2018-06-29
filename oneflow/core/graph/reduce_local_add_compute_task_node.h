#ifndef ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceLocalAddCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceLocalAddCompTaskNode);
  ReduceLocalAddCompTaskNode() = default;
  ~ReduceLocalAddCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceLocalAdd; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;

  int64_t min_in_parallel_id_;
  int64_t min_out_parallel_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_LOCAL_ADD_COMPUTE_TASK_NODE_H_
