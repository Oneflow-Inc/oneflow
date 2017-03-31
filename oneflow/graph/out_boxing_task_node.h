#ifndef ONEFLOW_GRAPH_OUT_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_OUT_BOXING_TASK_NODE_H_

#include "graph/boxing_task_node.h"

namespace oneflow {

class OutBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutBoxingTaskNode);
  OutBoxingTaskNode() = default;
  ~OutBoxingTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new OutBoxingTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    BoxingTaskNode::InitWithFwNode(fw_node);
  }
  void FwBuildExecGraph() override;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_OUT_BOXING_TASK_NODE_H_
