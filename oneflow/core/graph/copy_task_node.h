#ifndef ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CopyTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

 protected:
  virtual std::shared_ptr<const Operator> AddOp() const = 0;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph*) override;
  void InferBlobDescInProducedRegsts(TaskGraph*) override;
};

class CopyHDTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDTaskNode);
  CopyHDTaskNode() = default;
  ~CopyHDTaskNode() = default;

  bool IsH2D() const {
    return ((IsFwInCopy() && IsFwNode()) || (IsFwOutCopy() && IsBpNode()));
  }
  bool IsD2H() const { return !IsH2D(); }

  bool IsFwInCopy() const { return is_fw_in_copy_; }
  bool IsFwOutCopy() const { return !is_fw_in_copy_; }
  void SetFwInCopy();
  void SetFwOutCopy();

  std::string VisualStr() const override {
    return TaskNode::VisualStr() + "CopyHD";
  }

  void ToProto(TaskProto* ret) const override { TaskNode::ToProto(ret); };

 private:
  std::shared_ptr<const Operator> AddOp() const override;

  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_in_copy_ = static_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CopyHDTaskNode>();
  }
  TaskType task_type() const override { return kCopyHdTask; }

  bool is_fw_in_copy_;
};

class CopyCommNetTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetTaskNode);
  CopyCommNetTaskNode() = default;
  ~CopyCommNetTaskNode() = default;

  std::string VisualStr() const override {
    return TaskNode::VisualStr() + "CommNet";
  }

  void ToProto(TaskProto* ret) const override { TaskNode::ToProto(ret); };

 private:
  std::shared_ptr<const Operator> AddOp() const override;
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CopyCommNetTaskNode>();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    set_stage_node(fw_node->SoleInEdge()->src_node()->stage_node());
    set_task_id();
  }
  TaskType task_type() const override { return kCopyCommNetTask; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
