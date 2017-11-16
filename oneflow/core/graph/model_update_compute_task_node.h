#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class MdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompTaskNode);
  MdUpdtCompTaskNode() = default;
  ~MdUpdtCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;

  void set_random_seed(uint32_t val) { random_seed_ = val; }
  TodoTaskType GetTaskType() const override { return TodoTaskType::kMdUpdt; }
  void ToProto(TodoTaskProto*) override;

 private:
  uint32_t random_seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
