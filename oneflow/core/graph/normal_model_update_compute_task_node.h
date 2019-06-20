#ifndef ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"

namespace oneflow {

class NormalMdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtCompTaskNode);
  NormalMdUpdtCompTaskNode() = default;
  ~NormalMdUpdtCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override {}
  void LockRegsts() override;
  void EnableMemSharingBetweenFirstInAndProcessedMdDiffRegst();

  void set_random_seed(uint32_t val) { random_seed_ = val; }
  TaskType GetTaskType() const override { return TaskType::kNormalMdUpdt; }
  void ToProto(TaskProto*) override;
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMdUpdt; }

 private:
  const NormalForwardCompTaskNode* GetForwardTaskNode() const;
  bool IsTrainable() const;
  void FixPackedBlobDescOfProducedRegst() override;
  void InferProducedDataRegstTimeShape() override;
  uint32_t random_seed_;
  int64_t related_init_model_task_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
