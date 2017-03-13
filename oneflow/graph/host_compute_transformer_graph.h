#ifndef ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/compute_transformer_graph.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

// HostCompTransfmNode: HostComputeTransformerNode
class HostCompTransfmNode final : public ComputeTransformerNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmNode);
  HostCompTransfmNode() = default;
  ~HostCompTransfmNode() = default;

  void Init() {
    ComputeTransformerNode::Init();
  }
  
  std::shared_ptr<const Operator> op() const {
    return op_;
  }
  std::shared_ptr<const Operator>& mutable_op() {
    return op_;
  }

 private:
  std::shared_ptr<const Operator> op_;
};

class HostCompTransfmEdge final : public ComputeTransformerEdge {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmEdge);
  HostCompTransfmEdge() = default;
  ~HostCompTransfmEdge() = default;

  void Init() {
    ComputeTransformerEdge::Init();
  }

  BlobDescriptor* blob_desc_ptr() const {
    return blob_desc_ptr_;
  }

  void set_blob_desc_ptr(BlobDescriptor* new_blob_desc_ptr) {
    blob_desc_ptr_ = new_blob_desc_ptr;
  }

 private:
  BlobDescriptor* blob_desc_ptr_;
};

class HostCompTransfmGraph final : public ComputeTransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmGraph);
  HostCompTransfmGraph() = default;
  ~HostCompTransfmGraph() = default;

  void Init(const HostCompTaskNode* task_node, bool job_has_bp) {
    ComputeTransformerGraph::Init();
    task_node_ = task_node;
    job_has_bp_ = job_has_bp;
  }

  void FwBuildGraph() {
    FwBuildFromUserOps();
    if (job_has_bp_) {
      FwAddCopyInOp();
    }
    FwAddCloneOp();
    UpdateStartAndStop();
  }

 private:
  void FwBuildFromUserOps();
  void FwAddCopyInOp();
  void FwAddCloneOp();

  HostCompTransfmNode* NewHostCompTransfmNode() {
    auto ret = new HostCompTransfmNode;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }
  HostCompTransfmEdge* NewHostCompTransfmEdge(BlobDescriptor* blob_desc_ptr) {
    auto ret = new HostCompTransfmEdge;
    ret->Init();
    RegisterEdge(ret);
    ret->set_blob_desc_ptr(blob_desc_ptr);
    return ret;
  }

  const HostCompTaskNode* task_node_;
  bool job_has_bp_;
  std::unordered_map<std::string, std::vector<HostCompTransfmNode*>> extern_in_lbn2consumers_;
  std::unordered_map<std::string, BlobDescriptor> produced_lbn2blob_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
