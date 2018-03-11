#ifndef ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/model_update_compute_task_node.h"

namespace oneflow {

class NormalMdUpdtCompTaskNode final : public MdUpdtCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtCompTaskNode);
  NormalMdUpdtCompTaskNode() = default;
  ~NormalMdUpdtCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kNormalMdUpdt; }

 private:
  std::shared_ptr<const Operator> ConstructModelUpdateOp(
      int32_t in_num) override;
  void BindInRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
