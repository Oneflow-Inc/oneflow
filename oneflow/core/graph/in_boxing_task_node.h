#ifndef ONEFLOW_CORE_GRAPH_IN_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_IN_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/boxing_task_node.h"

namespace oneflow {

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<InBoxingTaskNode>();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    BoxingTaskNode::InitWithFwNode(fw_node);
  }
  void FwVirtualBuild() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_IN_BOXING_TASK_NODE_H_
