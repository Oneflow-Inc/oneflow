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
  
  std::string VisualStr() const override {
    return TaskNode::VisualStr() + "CopyHD";
  }
  
 private:
  void InitWithFwNode(TaskNode* fw_node) override;
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CopyHDTaskNode> ();
  }

  void FwBuildExecAndEnrollLbn2Regsts(TaskGraph*) override;
  void FwInferShape4LbnInProducedRegsts(TaskGraph*) override;
  void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) override;
  void BpInferShape4LbnInProducedRegsts(TaskGraph*) override;

  void CopyHdBuildExecAndEnrollLbn2Regsts();
  void CopyHdInferShape4LbnInProducedRegsts();

  bool is_fw_in_copy_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COPY_HD_TASK_NODE_H_
