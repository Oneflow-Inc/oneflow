#ifndef ONEFLOW_GRAPH_COPY_HD_TASK_NODE_H_
#define ONEFLOW_GRAPH_COPY_HD_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class CopyHDTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDTaskNode);
  CopyHDTaskNode() = default;
  ~CopyHDTaskNode() = default;
  
  bool IsH2D() const {
    return ((IsFwInCopy() && IsFwNode()) || (IsFwOutCopy() && IsBpNode()));
  }
  bool IsD2H() const {
    return !IsH2D();
  }

  bool IsFwInCopy() const { return is_fw_in_copy_; }
  bool IsFwOutCopy() const { return !is_fw_in_copy_; }
  void SetFwInCopy();
  void SetFwOutCopy();
  
  const std::vector<std::string>& CopiedLbns() const;
  
 private:
  void InitWithFwNode(TaskNode* fw_node) override;

  void FwBuildExecAndProducedRegsts(Path*) override { TODO(); }
  void BpBuildExecAndProducedRegsts(Path*) override { TODO(); }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new CopyHDTaskNode);
  }

  bool is_fw_in_copy_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COPY_HD_TASK_NODE_H_
