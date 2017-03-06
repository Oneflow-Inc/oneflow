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

  void Init() {
    Node::Init();
    // struct style
  }

  const MachineId& machine_id() const { return machine_id_; }
  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  
  MachineId& mutable_machine_id() { return machine_id_; }
  ThreadLocalId& mutable_thread_local_id() { return thread_local_id_; }

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

  void Init() {
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

 private:
};

// CommNet: Communication Network
class CommNetTnd : public TaskNode {
 public:
  DISALLOW_COPY_AND_MOVE(CommNetTnd);
  CommNetTnd() = default;
  ~CommNetTnd() = default;

  void Init() {
    TaskNode::Init();
  }

 private:
};

class TaskGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = default;
  ~TaskGraph() = default;
  
  void Init(const StageGraph* stage_dag,
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
  
  HostComputeTnd* NewHostComputeTnd();
  DeviceComputeTnd* NewDeviceComputeTnd();
  CopyHDTnd* NewCopyHDTnd();
  BoxingTnd* NewBoxingTnd();
  CommNetTnd* NewCommNetTnd();

  void InitComputeTnds(const StageGraph* stage_dag,
                       const IDMap& id_map,
                       Stage2TndsMap* stage2pons);
  void Stage2DeviceComputeTnds(const StageNode* stage,
                               const IDMap& id_map,
                               TndsWithinStage* pons_within_stage,
                               bool is_first_stage,
                               bool is_last_stage);
  void Stage2HostComputeTnds(const StageNode* stage,
                             const IDMap& id_map,
                             TndsWithinStage* pons_within_stage);
  void InitBoxingTnds(const StageGraph* stage_dag,
                      const IDMap& id_map,
                      Stage2TndsMap* stage2pons);
  void InitInboxingTnd(const StageNode* stage,
                       const IDMap& id_map,
                       TndsWithinStage* pons_within_stage);
  void InitOutBoxingTnd(const StageNode* stage,
                        const IDMap& id_map,
                        TndsWithinStage* pons_within_stage);
  void ConnectTnds(const StageGraph* stage_dag,
                   const Stage2TndsMap* stage2pons);
  void GenerateBpNodes();

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
