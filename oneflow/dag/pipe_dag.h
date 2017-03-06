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

class PipeNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(PipeNode);
  PipeNode() = default;
  ~PipeNode() = default;

  void Init() {
    DagNode::Init();
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

// Pn: PipeNode
class ComputePn : public PipeNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputePn);
  ComputePn() = default;
  ~ComputePn() = default;

  void Init() {
    PipeNode::Init();
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

class HostComputePn : public ComputePn {
 public:
  DISALLOW_COPY_AND_MOVE(HostComputePn);
  HostComputePn() = default;
  ~HostComputePn() = default;

  void Init() {
    ComputePn::Init();
  }

 private:

};

class DeviceComputePn : public ComputePn {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceComputePn);
  DeviceComputePn() = default;
  ~DeviceComputePn() = default;
  
  void Init() {
    ComputePn::Init();
  }

 private:
};

// HD: Host and Device
class CopyHDPn : public PipeNode {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDPn);
  CopyHDPn() = default;
  ~CopyHDPn() = default;
  
  void Init() {
    PipeNode::Init();
  }

 private:
};

class BoxingPn : public PipeNode {
 public:
  DISALLOW_COPY_AND_MOVE(BoxingPn);
  BoxingPn() = default;
  ~BoxingPn() = default;
  
  void Init() {
    PipeNode::Init();
  }

 private:
};

// CommNet: Communication Network
class CommNetPn : public PipeNode {
 public:
  DISALLOW_COPY_AND_MOVE(CommNetPn);
  CommNetPn() = default;
  ~CommNetPn() = default;

  void Init() {
    PipeNode::Init();
  }

 private:
};

class PipeDag : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(PipeDag);
  PipeDag() = default;
  ~PipeDag() = default;
  
  void Init(const StageDag* stage_dag,
            const IDMap& id_map,
            bool need_bp);

 private:
  struct PnsWithinStage {
    std::vector<PipeNode*> compute_in_pns;
    std::vector<PipeNode*> compute_out_pns;
    BoxingPn* in_boxing_pn;
    BoxingPn* out_boxing_pn;
  };
  
  using Stage2PnsMap =
    std::unordered_map<const StageNode*, PnsWithinStage>;
  
  HostComputePn* NewHostComputePn();
  DeviceComputePn* NewDeviceComputePn();
  CopyHDPn* NewCopyHDPn();
  BoxingPn* NewBoxingPn();
  CommNetPn* NewCommNetPn();

  void InitComputePns(const StageDag* stage_dag,
                      const IDMap& id_map,
                      Stage2PnsMap* stage2pons);
  void Stage2DeviceComputePns(const StageNode* stage,
                              const IDMap& id_map,
                              PnsWithinStage* pons_within_stage,
                              bool is_first_stage,
                              bool is_last_stage);
  void Stage2HostComputePns(const StageNode* stage,
                            const IDMap& id_map,
                            PnsWithinStage* pons_within_stage);
  void InitBoxingPns(const StageDag* stage_dag,
                     const IDMap& id_map,
                     Stage2PnsMap* stage2pons);
  void InitInboxingPn(const StageNode* stage,
                      const IDMap& id_map,
                      PnsWithinStage* pons_within_stage);
  void InitOutBoxingPn(const StageNode* stage,
                       const IDMap& id_map,
                       PnsWithinStage* pons_within_stage);
  void ConnectPns(const StageDag* stage_dag,
                  const Stage2PnsMap* stage2pons);
  void GenerateBpNodes();

};

} // namespace oneflow

#endif // ONEFLOW_DAG_PIPE_DAG_H_
