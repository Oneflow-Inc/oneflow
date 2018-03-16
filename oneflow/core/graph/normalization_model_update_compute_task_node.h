#ifndef ONEFLOW_CORE_GRAPH_NORMALIZATION_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMALIZATION_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalizationMdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationMdUpdtCompTaskNode);
  NormalizationMdUpdtCompTaskNode() = default;
  ~NormalizationMdUpdtCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override;
  void LockRegsts() override {}

  TaskType GetTaskType() const override {
    return TaskType::kNormalizationMdUpdt;
  }
  void ToProto(TaskProto*) override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMALIZATION_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
