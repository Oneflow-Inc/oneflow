#ifndef ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalForwardCompTaskNode);
  NormalForwardCompTaskNode() = default;
  ~NormalForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  bool IsReadyForBuild() override;

  TaskType GetTaskType() const override { return TaskType::kNormalForward; }
  bool HasBackwardCompTaskNode();
  virtual void ToProto(TaskProto*) override;

  void set_random_seed(int64_t random_seed) { random_seed_ = random_seed; }

 private:
  void BuildExecGphAndRegst() override;
  void LockRegsts() override;
  void BuildExecGphStructAndBindInRegst();
  void BuildOutRegst();
  void BuildActivationRegst();
  void BuildModel7ConstModel7DataTmp7BufRegsts();
  void InferProducedDataRegstTimeShape() override;

  int64_t random_seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
