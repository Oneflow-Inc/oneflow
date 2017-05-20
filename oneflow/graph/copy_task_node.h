#ifndef ONEFLOW_COPY_TASK_NODE_H_
#define ONEFLOW_COPY_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class CopyTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

 protected:
  virtual std::shared_ptr<const Operator> ConstructOp() const = 0;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph*) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph*) override;

};

class CopyHDTaskNode final : public CopyTaskNode {
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
  
  void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    ret->set_type(TaskType::CopyHdTask);
    ret->set_is_h2d(IsH2D());
  };

 private:
  std::shared_ptr<const Operator> ConstructOp() const override;

  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CopyHDTaskNode> ();
  }

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

  void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    ret->set_type(TaskType::CopyCommNetTask);
  };

 private:
  std::shared_ptr<const Operator> ConstructOp() const override;
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<CopyCommNetTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }

};

} // namespace oneflow

#endif // ONEFLOW_COPY_TASK_NODE_H_
