#ifndef ONEFLOW_DAG_PIPE_DAG_H_
#define ONEFLOW_DAG_PIPE_DAG_H_

#include "dag/dag.h"
#include "common/id_map.h"

namespace oneflow {

class PipeDataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(PipeDataNode);
  PipeDataNode() = default;
  ~PipeDataNode() = default();

  void Init() {
    DagNode::Init();
  }
    
 private:
};

class PipeOpNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(PipeOpNode);
  PipeOpNode() = default;
  ~PipeOpNode() = default;

  void Init() {
    DagNode::Init();
  }

 private:
};

// Pon: PipeOpNode
class HostComputePon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostComputePon);
  HostComputePon() = default;
  ~HostComputePon() = default;

  void Init() {
    PipeOpNode::Init();
  }

 private:
};

class DeviceComputePon : public PipeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceComputePon);
  DeviceComputePon() = default;
  ~DeviceComputePon() = default;
  
  void Init() {
    PipeOpNode::Init();
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
  HostComputePon* NewHostComputePon();
  DeviceComputePon* NewDeviceComputePon();
  CopyHDPon* NewCopyHDPon();
  CommNetPon* NewCommNetPon();

};

} // namespace oneflow

#endif // ONEFLOW_DAG_PIPE_DAG_H_
