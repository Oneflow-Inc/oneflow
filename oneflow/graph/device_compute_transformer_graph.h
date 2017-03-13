#ifndef ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_

namespace oneflow {

class DeviceCompTransfmNode final : public ComputeTransformerNode {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceCompTransfmNode);
  DeviceCompTransfmNode() = default;
  ~DeviceCompTransfmNode() = default;

  void Init() {
    ComputeTransformerNode::Init();
  }

 private:
};

class DeviceCompTransfmEdge final : public ComputeTransformerEdge {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceCompTransfmEdge);
  DeviceCompTransfmEdge() = default;
  ~DeviceCompTransfmEdge() = default;

  void Init() {
    ComputeTransformerEdge::Init();
  }
 
 private:
};

class DeviceCompTransfmGraph final : public ComputeTransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceCompTransfmGraph);
  DeviceCompTransfmGraph() = default;
  ~DeviceCompTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    ComputeTransformerGraph::Init(task_node, job_has_bp);
  }

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

  TransfmNode* NewTransfmNode() override {
    auto ret = new DeviceCompTransfmNode;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }

  TransfmEdge* NewTransfmEdge(BlobDescriptor* blob_desc_ptr) override {
    auto ret = new DeviceCompTransfmEdge;
    ret->Init();
    RegisterEdge(ret);
    ret->set_blob_desc_ptr(blob_desc_ptr);
    return ret;
  }
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
