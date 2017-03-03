#ifndef ONEFLOW_DAG_PIPE_DAG_H_
#define ONEFLOW_DAG_PIPE_DAG_H_

#include "dag/stage_dag.h"
#include "layer/base_layer_desc.h"
#include "job/parallel_desc.h"
#include "common/id_map.h"

namespace oneflow {

class PipeDataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(PipeDataNode);
  PipeDataNode() = default;
  ~PipeDataNode() = default;

  void Init() {
    DagNode::Init();
  }
    
 private:
};

class PipeOpNode : public OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(PipeOpNode);
  PipeOpNode() = default;
  ~PipeOpNode() = default;

  void Init() {
    OpNode::Init();
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

// Pon: PipeOpNode
class ComputePon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputePon);
  ComputePon() = default;
  ~ComputePon() = default;

  void Init() {
    PipeOpNode::Init();
  }
  
  const std::vector<std::shared_ptr<const BaseLayerDesc>>& layer_desc_vec() const {
    return layer_desc_vec_;
  }
  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  const std::shared_ptr<const ParallelDesc>& parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  
  std::vector<std::shared_ptr<const BaseLayerDesc>>& mutable_layer_desc_vec() {
    return layer_desc_vec_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

 private:
  std::vector<std::shared_ptr<const BaseLayerDesc>> layer_desc_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class HostComputePon : public ComputePon {
 public:
  DISALLOW_COPY_AND_MOVE(HostComputePon);
  HostComputePon() = default;
  ~HostComputePon() = default;

  void Init() {
    ComputePon::Init();
  }

 private:

};

class DeviceComputePon : public ComputePon {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceComputePon);
  DeviceComputePon() = default;
  ~DeviceComputePon() = default;
  
  void Init() {
    ComputePon::Init();
  }

 private:
};

// HD: Host and Device
class CopyHDPon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDPon);
  CopyHDPon() = default;
  ~CopyHDPon() = default;
  
  void Init() {
    PipeOpNode::Init();
  }

 private:
};

class BoxingPon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(BoxingPon);
  BoxingPon() = default;
  ~BoxingPon() = default;
  
  void Init() {
    PipeOpNode::Init();
  }

 private:
};

// CommNet: Communication Network
class CommNetPon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(CommNetPon);
  CommNetPon() = default;
  ~CommNetPon() = default;

  void Init() {
    PipeOpNode::Init();
  }

 private:
};

class PipeDag : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(PipeDag);
  PipeDag() = default;
  ~PipeDag() = default;
  
  void Init(std::shared_ptr<const StageDag> stage_dag,
            const IDMap& id_map);

 private:
  void ConnectTwoOp(PipeOpNode* predecessor, PipeOpNode* successor) {
    PipeDataNode* data_node = NewPipeDataNode();
    data_node->AddPredecessor(predecessor);
    successor->AddPredecessor(data_node);
  }
  PipeDataNode* NewPipeDataNode();
  HostComputePon* NewHostComputePon();
  DeviceComputePon* NewDeviceComputePon();
  CopyHDPon* NewCopyHDPon();
  BoxingPon* NewBoxingPon();
  CommNetPon* NewCommNetPon();

  struct PonsWithinStage {
    std::vector<PipeOpNode*> compute_in_pons;
    std::vector<PipeOpNode*> compute_out_pons;
    BoxingPon* in_boxing_pon;
    BoxingPon* out_boxing_pon;
  };
  
  using Stage2PonsMap =
    std::unordered_map<const StageOpNode*, PonsWithinStage>;

  void InitComputePons(const StageDag* stage_dag,
                       const IDMap& id_map,
                       Stage2PonsMap* stage2pons);
  void Stage2DeviceComputePons(const StageOpNode* stage_op,
                               const IDMap& id_map,
                               PonsWithinStage* pons_within_stage,
                               bool is_first_stage,
                               bool is_last_stage);
  void Stage2HostComputePons(const StageOpNode* stage_op,
                             const IDMap& id_map,
                             PonsWithinStage* pons_within_stage);
  void InitBoxingPons(const StageDag* stage_dag,
                      const IDMap& id_map,
                      Stage2PonsMap* stage2pons);
  void InitInboxingPon(const StageOpNode* stage_op,
                       const IDMap& id_map,
                       PonsWithinStage* pons_within_stage);
  void InitOutBoxingPon(const StageOpNode* stage_op,
                        const IDMap& id_map,
                        PonsWithinStage* pons_within_stage);
  void ConnectPons(const StageDag* stage_dag,
                   const Stage2PonsMap* stage2pons);

};

} // namespace oneflow

#endif // ONEFLOW_DAG_PIPE_DAG_H_
