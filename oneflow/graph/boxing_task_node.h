#ifndef ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_BOXING_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class BoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  ~BoxingTaskNode() = default;

  bool IsFwInBoxing() const { return is_fw_in_boxing_; }
  bool IsFwOutBoxing() const { return !is_fw_in_boxing_; }
  void SetFwInBoxing();
  void SetFwOutBoxing();

  void BuildExecGraphAndSetRegisterDescs() override;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new BoxingTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override;
  
  bool is_fw_in_boxing_;
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
