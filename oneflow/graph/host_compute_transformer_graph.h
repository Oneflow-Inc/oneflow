#ifndef ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/compute_transformer_graph.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class HostCompTransfmNode final : public CompTransfmNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmNode);
  HostCompTransfmNode() = default;
  ~HostCompTransfmNode() = default;

  void Init() {
    CompTransfmNode::Init();
  }

 private:
};

class HostCompTransfmEdge final : public CompTransfmEdge {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmEdge);
  HostCompTransfmEdge() = default;
  ~HostCompTransfmEdge() = default;

  void Init() {
    CompTransfmEdge::Init();
  }

 private:
};

class HostCompTransfmGraph final : public CompTransfmGraph {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmGraph);
  HostCompTransfmGraph() = default;
  ~HostCompTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    ComputeTransformerGraph::Init(task_node, job_has_bp);
  }

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::H2H;
  }

  TransfmNode* NewTransfmNode() override {
    auto ret = new HostCompTransfmNode;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }

  TransfmEdge* NewTransfmEdge(BlobDescriptor* blob_desc_ptr) override {
    auto ret = new HostCompTransfmEdge;
    ret->Init();
    RegisterEdge(ret);
    ret->set_blob_desc_ptr(blob_desc_ptr);
    return ret;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
