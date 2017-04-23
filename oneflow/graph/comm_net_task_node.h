#ifndef ONEFLOW_GRAPH_COMM_NET_TASK_NODE_H_
#define ONEFLOW_GRAPH_COMM_NET_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class CommNetTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetTaskNode);
  CommNetTaskNode() = default;
  ~CommNetTaskNode() = default;

  bool IsSender() const {
    return (IsFwNode() && is_fw_sender_)
        || (IsBpNode() && !is_fw_sender_);
  }
  bool IsReceiver() const {
    return !IsSender();
  }

  void SetFwSender() {
    CHECK(IsFwNode());
    is_fw_sender_ = true;
  }
  void SetFwReceiver() {
    CHECK(IsFwNode());
    is_fw_sender_ = false;
  }

 private:
  void FwBuildExecAndProducedRegsts(TaskGraph*) override;
  void BpBuildExecAndProducedRegsts(TaskGraph*) override;

  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CommNetTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_sender_ = of_dynamic_cast<CommNetTaskNode*>(fw_node)->is_fw_sender_;
  }

  bool is_fw_sender_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMM_NET_TASK_NODE_H_
