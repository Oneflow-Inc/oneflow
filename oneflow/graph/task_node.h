#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "graph/stage_graph.h"
#include "graph/register_desc.h"

namespace oneflow {

class TransfmGraph;
class TaskEdge;

class TaskNode : public Node<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode() {
    stage_node_ = nullptr;
    related_fw_or_bp_node_ = nullptr;
    transfm_graph_ = nullptr;
  }
  virtual ~TaskNode() = default;

  // Is fw or bp
  bool IsFwNode() const { return is_fw_node_; }
  bool IsBpNode() const { return !is_fw_node_; }
  void SetFwNode() { is_fw_node_ = true; }

  // chain_node

  const ChainNode* chain_node() const {
    return stage_node_->chain_node();
  }

  // stage_node_
  const StageNode* stage_node() const {
    return stage_node_;
  }
  void set_stage_node(const StageNode* new_stage_node) {
    CHECK(IsFwNode());
    stage_node_ = new_stage_node;
  }
  
  // thread_local_id_
  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  ThreadLocalId& mut_thread_local_id() {
    CHECK(IsFwNode());
    return thread_local_id_;
  }

  // Get related fw/bp node
  TaskNode* GetFwNode() const {
    CHECK(IsBpNode());
    return related_fw_or_bp_node_;
  }
  TaskNode* GetBpNode() const {
    CHECK(IsFwNode());
    return related_fw_or_bp_node_;
  }

  // transfm_graph
  TransfmGraph* transfm_graph() const {
    return transfm_graph_;
  }

  // Functions about Build BP
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const;
  virtual void InitWithFwNode(TaskNode* fw_node);

 private:
  const StageNode* stage_node_;
  ThreadLocalId thread_local_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  TransfmGraph* transfm_graph_;

};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() { register_desc_ = nullptr; }
  ~TaskEdge() = default;
  
  RegisterDesc* register_desc() const {
    return register_desc_;
  }
  void set_register_desc(RegisterDesc* new_ptr) {
    register_desc_ = new_ptr;
  }

 private:
  RegisterDesc* register_desc_;

};

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  bool HasOpWithOutDiff() const;
  bool HasOpWithIndiff() const;

  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }

 private:

};

class HostCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new HostCompTaskNode);
  }

  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

 private:

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new DeviceCompTaskNode);
  }

  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

 private:
};

class CopyHDTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDTaskNode);
  CopyHDTaskNode() = default;
  ~CopyHDTaskNode() = default;
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new CopyHDTaskNode);
  }

  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_in_copy_ =
        of_dynamic_cast<const CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
  }

  bool IsH2D() const {
    return ((IsFwInCopy() && IsFwNode()) || (IsFwOutCopy() && IsBpNode()));
  }
  bool IsD2H() const {
    return !IsH2D();
  }

  const std::vector<std::string>& RelatedLbns() const {
    if (IsFwInCopy()) {
      return stage_node()->chain_node()->input_lbns();
    } else {
      return stage_node()->chain_node()->output_lbns();
    }
  }

  bool IsFwInCopy() const { return is_fw_in_copy_; }
  bool IsFwOutCopy() const { return !is_fw_in_copy_; }
  void SetFwInCopy() {
    CHECK(IsFwNode());
    is_fw_in_copy_ = true;
  }
  void SetFwOutCopy() {
    CHECK(IsFwNode());
    is_fw_in_copy_ = false;
  }

 private:
  bool is_fw_in_copy_;

};

class BoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  ~BoxingTaskNode() = default;
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new BoxingTaskNode);
  }

  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_in_boxing_ =
        of_dynamic_cast<const BoxingTaskNode*>(fw_node)->is_fw_in_boxing_;
  }

  bool IsFwInBoxing() const { return is_fw_in_boxing_; }
  bool IsFwOutBoxing() const { return !is_fw_in_boxing_; }
  void SetFwInBoxing() {
    CHECK(IsFwNode());
    is_fw_in_boxing_ = true;
  }
  void SetFwOutBoxing() {
    CHECK(IsFwNode());
    is_fw_in_boxing_ = false;
  }

 private:
  bool is_fw_in_boxing_;
};

class CommNetTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetTaskNode);
  CommNetTaskNode() = default;
  ~CommNetTaskNode() = default;

  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new CommNetTaskNode);
  }

  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_sender_ =
        of_dynamic_cast<const CommNetTaskNode*>(fw_node)->is_fw_sender_;
  }

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
  bool is_fw_sender_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
