#ifndef ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalMdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtCompTaskNode);
  NormalMdUpdtCompTaskNode() = default;
  ~NormalMdUpdtCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override;
  void LockRegsts() override;

  void set_random_seed(uint32_t val) { random_seed_ = val; }
  TaskType GetTaskType() const override { return TaskType::kNormalMdUpdt; }
  void ToProto(TaskProto*) override;

 private:
  uint32_t random_seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
