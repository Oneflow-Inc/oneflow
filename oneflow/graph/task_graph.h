#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "graph/stage_graph.h"
#include "operator/operator.h"
#include "job/parallel_desc.h"
#include "common/id_map.h"

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

  bool HasOpWithOutDiff() const {
    for (std::shared_ptr<const Operator> op : op_vec_) {
      if (! op->data_blob_desc_set().output_diff_blob_names().empty()) {
        return true;
      }
    }
    return false;
  }
  bool HasOpWithIndiff() const {
    for (std::shared_ptr<const Operator> op : op_vec_) {
      if (! op->data_blob_desc_set().input_diff_blob_names().empty()) {
        return true;
      }
    }
    return false;
  }
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

class TaskEdge final : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() = default;
  
  void Init() {
    Edge::Init();
  }

 private:
};

class TaskGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = default;
  ~TaskGraph() = default;
  
  void Init(const StageGraph* stage_graph,
            const IDMap& id_map,
            bool need_bp);

 private:
  struct TndsWithinStage {
    std::vector<TaskNode*> compute_in_tnds;
    std::vector<TaskNode*> compute_out_tnds;
    BoxingTnd* in_boxing_tnd;
    BoxingTnd* out_boxing_tnd;
  };
  
  using Stage2TndsMap =
    std::unordered_map<const StageNode*, TndsWithinStage>;
  
  template<typename TaskNodeType>
  TaskNodeType* NewTaskNode() {
    static_assert(std::is_base_of<TaskNode, TaskNodeType>::value, "");
    TaskNodeType* ret = new TaskNodeType;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }

  TaskEdge* NewTaskEdge() {
    TaskEdge* ret = new TaskEdge;
    ret->Init();
    RegisterEdge(ret);
    return ret;
  }

  TaskNode* ConstructBpNode(TaskNode* fw_node) {
    std::unique_ptr<TaskNode> node = fw_node->CloneWithOnlyTaskProperty();
    TaskNode* ret = node.get();
    RegisterNode(std::move(node));
    return ret;
  }
  
  void InitComputeTnds(const StageGraph* stage_graph,
                       const IDMap& id_map,
                       Stage2TndsMap* stage2tnds);
  void Stage2DeviceComputeTnds(const StageNode* stage,
                               const IDMap& id_map,
                               TndsWithinStage* tnds_within_stage,
                               bool is_first_stage,
                               bool is_last_stage);
  void Stage2HostComputeTnds(const StageNode* stage,
                             const IDMap& id_map,
                             TndsWithinStage* tnds_within_stage);
  void InitBoxingTnds(const StageGraph* stage_graph,
                      const IDMap& id_map,
                      Stage2TndsMap* stage2tnds);
  void InitInboxingTnd(const StageNode* stage,
                       const IDMap& id_map,
                       TndsWithinStage* tnds_within_stage);
  void InitOutBoxingTnd(const StageNode* stage,
                        const IDMap& id_map,
                        TndsWithinStage* tnds_within_stage);
  void ConnectTnds(const StageGraph* stage_graph,
                   const Stage2TndsMap* stage2tnds);
  void GenerateRelatedBpNodes(
      std::function<void(const TaskNode*, TaskNode*)> add_fw_bp_pair,
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      std::vector<TaskNode*> *turning_node_vec);
  void BackwardConnect(
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      const std::unordered_map<TaskNode*, const TaskNode*>& bp_node2fw_node,
      const std::vector<TaskNode*>& turning_node_vec);
  void BuildBpStruct();

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
