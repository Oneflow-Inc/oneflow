#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "graph/stage_graph.h"
#include "graph/host_comp_transfm_graph.h"
#include "graph/device_comp_transfm_graph.h"
#include "graph/boxing_transfm_graph.h"
#include "graph/copy_hd_transfm_graph.h"
#include "graph/comm_net_transfm_graph.h"
#include "graph/register_desc.h"

namespace oneflow {

class TaskEdge;

class TaskNode : public Node<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode();
  virtual ~TaskNode() = default;

  // Getters
  bool IsFwNode() const { return is_fw_node_; }
  bool IsBpNode() const { return !is_fw_node_; }
  const ChainNode* chain_node() const { return stage_node_->chain_node();}
  const StageNode* stage_node() const { return stage_node_; }
  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  TransfmGraph* transfm_graph() const { return transfm_graph_.get(); }
  TaskNode* GetFwNode() const;
  TaskNode* GetBpNode() const;
  
  // Setters
  void SetFwNode() { is_fw_node_ = true; }
  void set_stage_node(const StageNode*);
  ThreadLocalId& mut_thread_local_id();

  //
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();
  void SetNewTransfmGraph();
 
 protected:
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const;
  virtual void InitWithFwNode(TaskNode* fw_node);
  virtual std::unique_ptr<TransfmGraph> CreateTransfmGraph() const;

 private:
  const StageNode* stage_node_;
  ThreadLocalId thread_local_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  std::unique_ptr<TransfmGraph> transfm_graph_;

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

 protected:
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

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new HostCompTaskNode);
  }
  std::unique_ptr<TransfmGraph> CreateTransfmGraph() const override {
    return std::unique_ptr<TransfmGraph> (new HostCompTransfmGraph);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new DeviceCompTaskNode);
  }
  std::unique_ptr<TransfmGraph> CreateTransfmGraph() const override {
    return std::unique_ptr<TransfmGraph> (new DeviceCompTransfmGraph);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

};

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

  const std::vector<std::string>& RelatedLbns() const;

  bool IsFwInCopy() const { return is_fw_in_copy_; }
  bool IsFwOutCopy() const { return !is_fw_in_copy_; }
  void SetFwInCopy();
  void SetFwOutCopy();

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new CopyHDTaskNode);
  }
  std::unique_ptr<TransfmGraph> CreateTransfmGraph() const override {
    return std::unique_ptr<TransfmGraph> (new CopyHDTransfmGraph);
  }
  void InitWithFwNode(TaskNode* fw_node) override;

  bool is_fw_in_copy_;

};

class BoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  ~BoxingTaskNode() = default;

  bool IsFwInBoxing() const { return is_fw_in_boxing_; }
  bool IsFwOutBoxing() const { return !is_fw_in_boxing_; }
  void SetFwInBoxing();
  void SetFwOutBoxing();

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new BoxingTaskNode);
  }
  std::unique_ptr<TransfmGraph> CreateTransfmGraph() const override {
    return std::unique_ptr<TransfmGraph> (new BoxingTransfmGraph);
  }
  void InitWithFwNode(TaskNode* fw_node) override;
  
  bool is_fw_in_boxing_;
};

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
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new CommNetTaskNode);
  }
  std::unique_ptr<TransfmGraph> CreateTransfmGraph() const override {
    return std::unique_ptr<TransfmGraph> (new CommNetTransfmGraph);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    is_fw_sender_ =
        of_dynamic_cast<const CommNetTaskNode*>(fw_node)->is_fw_sender_;
  }

  bool is_fw_sender_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
