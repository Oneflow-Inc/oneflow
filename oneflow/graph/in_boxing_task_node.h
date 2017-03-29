#ifndef ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_

#include "graph/boxing_task_node.h"

namespace oneflow {

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

 private:
  void FwBuildExecGraphAndSetProducedRegisterDescs() override {
    LOG(FATAL) << "TODO";
  }
  void BpBuildExecGraphAndSetProducedRegisterDescs() override {
    LOG(FATAL) << "TODO";
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new InBoxingTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    BoxingTaskNode::InitWithFwNode(fw_node);
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_
