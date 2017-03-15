#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "graph/stage_graph.h"

namespace oneflow {

class TaskNode : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode() = default;
  virtual ~TaskNode() = default;

  virtual void Init() {
    Node::Init();
  }
  
  // Getters and Setters
  const StageNode* stage_node() const {
    return stage_node_;
  }
  void set_stage_node(const StageNode* new_stage_node) {
    stage_node_ = new_stage_node;
  }

  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  ThreadLocalId& mutable_thread_local_id() { return thread_local_id_; }
  
  bool IsFwNode() const { return is_fw_node_; }
  bool IsBpNode() const { return !is_fw_node_; }
  void SetFwNode() { is_fw_node_ = true; }
  void SetBpNode() { is_fw_node_ = false; }

  // Clone without Node's property
  std::unique_ptr<TaskNode> CloneWithOnlyTaskProperty() const {
    std::unique_ptr<TaskNode> new_node  = CreateSameTypeNode();
    new_node->CopyWithOnlyTaskProperty(*this);
    return new_node;
  }

  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const = 0;

  virtual void CopyWithOnlyTaskProperty(const TaskNode& rhs) {
    stage_node_ = rhs.stage_node_;
    thread_local_id_ = rhs.thread_local_id_;
    is_fw_node_ = rhs.is_fw_node_;
  }

 private:
  const StageNode* stage_node_;
  ThreadLocalId thread_local_id_;
  bool is_fw_node_;

};

class CompTaskNode : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  virtual void Init() {
    TaskNode::Init();
  }
  
  bool HasOpWithOutDiff() const;
  bool HasOpWithIndiff() const;

  virtual void CopyWithOnlyTaskProperty(const CompTaskNode& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:

};

class HostCompTaskNode final : public CompTaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

  void Init() {
    CompTaskNode::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new HostCompTaskNode);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const HostCompTaskNode& rhs) {
    CompTaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
  void Init() {
    CompTaskNode::Init();
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new DeviceCompTaskNode);
    new_node->Init();
    return new_node;
  }
 
  void CopyWithOnlyTaskProperty(const DeviceCompTaskNode& rhs) {
    CompTaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

// HD: Host and Device
class CopyHDTaskNode final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDTaskNode);
  CopyHDTaskNode() = default;
  ~CopyHDTaskNode() = default;
  
  void Init() {
    TaskNode::Init();
  }

  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new CopyHDTaskNode);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const CopyHDTaskNode& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
    is_in_copy_ = rhs.is_in_copy_;
  }

  bool IsH2D() const {
    return ((IsInCopy() && IsFwNode()) || (IsOutCopy() && IsBpNode()));
  }
  bool IsD2H() const {
    return !IsH2D();
  }

  bool IsInCopy() const { return is_in_copy_; }
  bool IsOutCopy() const { return !is_in_copy_; }
  void SetInCopy() { is_in_copy_ = true; }
  void SetOutCopy() { is_in_copy_ = false; }

 private:
  bool is_in_copy_;

};

class BoxingTaskNode final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  ~BoxingTaskNode() = default;
  
  void Init() {
    TaskNode::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new BoxingTaskNode);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const BoxingTaskNode& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
    is_in_boxing_ = rhs.is_in_boxing_;
  }

  bool IsInBoxing() const { return is_in_boxing_; }
  bool IsOutBoxing() const { return !is_in_boxing_; }
  void SetInBoxing() { is_in_boxing_ = true; }
  void SetOutBoxing() { is_in_boxing_ = false; }

 private:
  bool is_in_boxing_;
};

// CommNet: Communication Network
class CommNetTaskNode final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CommNetTaskNode);
  CommNetTaskNode() = default;
  ~CommNetTaskNode() = default;

  void Init() {
    TaskNode::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new CommNetTaskNode);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const CommNetTaskNode& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
