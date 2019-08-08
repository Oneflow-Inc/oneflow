#ifndef ONEFLOW_CORE_GRAPH_CUDA_RING_ALL_REDUCE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_CUDA_RING_ALL_REDUCE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class CudaRingAllReduceCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceCompTaskNode);
  CudaRingAllReduceCompTaskNode() = default;
  ~CudaRingAllReduceCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kCudaRingAllReduce; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;
  void SetRecvSendNodes(const std::vector<TaskNode*>& recv_from,
                        const std::vector<TaskNode*>& send_to);

 private:
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  void FixSendRegstMemCase(MemoryCase* mem_case, TaskNode* send_to);

  std::vector<TaskNode*> send_to_;
  std::vector<TaskNode*> recv_from_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CUDA_RING_ALL_REDUCE_COMPUTE_TASK_NODE_H_
