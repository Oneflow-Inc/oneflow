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

// TaskNode: TaskNode
class CompTaskNode : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  virtual void Init() {
    TaskNode::Init();
  }
  
  const std::vector<std::shared_ptr<const Operator>>& op_vec() const {
    return *op_vec_ptr_;
  }
  std::vector<std::shared_ptr<const Operator>>& mutable_op_vec() {
    return *op_vec_ptr_;
  }
  std::shared_ptr<std::vector<std::shared_ptr<const Operator>>> op_vec_ptr() const {
    return op_vec_ptr_;
  }
  std::shared_ptr<std::vector<std::shared_ptr<const Operator>>>& mutable_op_vec_ptr() {
    return op_vec_ptr_;
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

  const std::vector<std::string>& input_lbns() const {
    return *input_lbns_ptr_;
  }
  std::vector<std::string>& mutable_input_lbns() {
    return *input_lbns_ptr_;
  }
  std::shared_ptr<std::vector<std::string>> input_lbns_ptr() const {
    return input_lbns_ptr_;
  }
  std::shared_ptr<std::vector<std::string>>& mutable_input_lbns_ptr() {
    return input_lbns_ptr_;
  }
  
  const std::vector<std::string>& output_lbns() const {
    return *output_lbns_ptr_;
  }
  std::vector<std::string>& mutable_output_lbns() {
    return *output_lbns_ptr_;
  }
  std::shared_ptr<std::vector<std::string>> output_lbns_ptr() const {
    return output_lbns_ptr_;
  }
  std::shared_ptr<std::vector<std::string>>& mutable_output_lbns_ptr() {
    return output_lbns_ptr_;
  }

  bool HasOpWithOutDiff() const;
  bool HasOpWithIndiff() const;

  virtual void CopyWithOnlyTaskProperty(const CompTaskNode& rhs) {
    TaskNode::CopyWithOnlyTaskProperty(rhs);
    op_vec_ptr_ = rhs.op_vec_ptr_;
    parallel_desc_ptr_ = rhs.parallel_desc_ptr_;
  }

 private:
  std::shared_ptr<std::vector<std::shared_ptr<const Operator>>> op_vec_ptr_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;
  std::shared_ptr<std::vector<std::string>> input_lbns_ptr_;
  std::shared_ptr<std::vector<std::string>> output_lbns_ptr_;

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
  }

 private:
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
  }

 private:
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
