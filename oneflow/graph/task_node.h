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
    // struct style
  }

  const MachineId& machine_id() const { return machine_id_; }
  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  
  MachineId& mutable_machine_id() { return machine_id_; }
  ThreadLocalId& mutable_thread_local_id() { return thread_local_id_; }

  // Clone without Node's property
  std::unique_ptr<TaskNode> CloneWithOnlyTaskProperty() const {
    std::unique_ptr<TaskNode> new_node  = CreateSameTypeNode();
    new_node->CopyWithOnlyTaskProperty(*this);
    return new_node;
  }

  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const = 0;

  virtual void CopyWithOnlyTaskProperty(const TaskNode& rhs) {
    machine_id_ = rhs.machine_id_;
    thread_local_id_ = rhs.thread_local_id_;
  }

 private:
  MachineId machine_id_;
  ThreadLocalId thread_local_id_;

};

// Tnd: TaskNode
class ComputeTnd : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeTnd);
  ComputeTnd() = default;
  virtual ~ComputeTnd() = default;

  virtual void Init() {
    TaskNode::Init();
  }
  
  const std::vector<std::shared_ptr<const Operator>>& op_vec() const {
    return op_vec_;
  }
  std::vector<std::shared_ptr<const Operator>>& mutable_op_vec() {
    return op_vec_;
  }

  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

  bool HasOpWithOutDiff() const;
  bool HasOpWithIndiff() const;

  virtual void CopyWithOnlyTaskProperty(const ComputeTnd& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
    op_vec_ = rhs.op_vec_;
    parallel_desc_ptr_ = rhs.parallel_desc_ptr_;
  }

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class HostComputeTnd final : public ComputeTnd {
 public:
  DISALLOW_COPY_AND_MOVE(HostComputeTnd);
  HostComputeTnd() = default;
  ~HostComputeTnd() = default;

  void Init() {
    ComputeTnd::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new HostComputeTnd);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const HostComputeTnd& rhs) {
    ComputeTnd::CopyWithOnlyTaskProperty(rhs);
  }

 private:

};

class DeviceComputeTnd final : public ComputeTnd {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceComputeTnd);
  DeviceComputeTnd() = default;
  ~DeviceComputeTnd() = default;
  
  void Init() {
    ComputeTnd::Init();
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new DeviceComputeTnd);
    new_node->Init();
    return new_node;
  }
 
  void CopyWithOnlyTaskProperty(const DeviceComputeTnd& rhs) {
    ComputeTnd::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

// HD: Host and Device
class CopyHDTnd final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDTnd);
  CopyHDTnd() = default;
  ~CopyHDTnd() = default;
  
  void Init() {
    TaskNode::Init();
  }

  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new CopyHDTnd);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const CopyHDTnd& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

class BoxingTnd final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(BoxingTnd);
  BoxingTnd() = default;
  ~BoxingTnd() = default;
  
  void Init() {
    TaskNode::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new BoxingTnd);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const BoxingTnd& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

// CommNet: Communication Network
class CommNetTnd final : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CommNetTnd);
  CommNetTnd() = default;
  ~CommNetTnd() = default;

  void Init() {
    TaskNode::Init();
  }
  
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    std::unique_ptr<TaskNode> new_node(new CommNetTnd);
    new_node->Init();
    return new_node;
  }

  void CopyWithOnlyTaskProperty(const CommNetTnd& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
