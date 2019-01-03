#ifndef ONEFLOW_CORE_GRAPH_NCCL_INTER_DEVICE_REDUCE_SUM_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_INTER_DEVICE_REDUCE_SUM_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NcclInterDeviceReduceSumForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclInterDeviceReduceSumForwardCompTaskNode);
  NcclInterDeviceReduceSumForwardCompTaskNode() = default;
  ~NcclInterDeviceReduceSumForwardCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kNcclInterDeviceReduceSumForward; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_INTER_DEVICE_REDUCE_SUM_FORWARD_COMPUTE_TASK_NODE_H_
