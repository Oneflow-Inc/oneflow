#ifndef ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_

#include "graph/compute_operator_graph.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class HostCompOpNode final : public ComputeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOpNode);
  HostCompOpNode() = default;
  ~HostCompOpNode() = default;

  void Init() {
    ComputeOpNode::Init();
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

class HostCompOpEdge final : public ComputeOpEdge {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOpEdge);
  HostCompOpEdge() = default;
  ~HostCompOpEdge() = default;

  void Init() {
    ComputeOpEdge::Init();
  }

  void set_blob_desc_ptr(BlobDescriptor* new_blob_desc_ptr) {
    blob_desc_ptr_ = new_blob_desc_ptr;
  }

 private:
  BlobDescriptor* blob_desc_ptr_;
};

class HostCompOperatorGraph final : public ComputeOperatorGraph {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOperatorGraph);
  HostCompOperatorGraph() = default;
  ~HostCompOperatorGraph() = default;

  void Init(const HostComputeTnd* task_node, bool job_has_bp) {
    ComputeOperatorGraph::Init();
    task_node_ = task_node;
    job_has_bp_ = job_has_bp;
  }

  void FwBuildGraph() {
    BuildFromUserOps();
    if (job_has_bp_) {
      AddCopyInOp();
    }
    AddCloneOp();
    UpdateStartAndStop();
  }

 private:
  void BuildFromUserOps();
  void AddCopyInOp();
  void AddCloneOp();


  HostCompOpNode* NewHostCompOpNode() {
    auto ret = new HostCompOpNode;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }
  HostCompOpEdge* NewHostCompOpEdge() {
    auto ret = new HostCompOpEdge;
    ret->Init();
    RegisterEdge(ret);
    return ret;
  }

  const HostComputeTnd* task_node_;
  bool job_has_bp_;
  std::unordered_map<std::string, HostCompOpNode*> input_lbn2consumer_;
  std::unordered_map<std::string, BlobDescriptor> produced_lbn2blob_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_
