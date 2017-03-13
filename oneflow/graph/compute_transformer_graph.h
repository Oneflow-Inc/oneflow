#ifndef ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/transformer_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CompTransfmNode : public TransfmNode {
 public:
  DISALLOW_COPY_AND_MOVE(CompTransfmNode);
  virtual ~CompTransfmNode() = default;

  virtual void Init() {
    TransfmNode::Init();
    // struct style
  }
 protected:
  CompTransfmNode() = default;

 private:

};

class CompTransfmEdge : public TransfmEdge {
 public:
  DISALLOW_COPY_AND_MOVE(CompTransfmEdge);
  virtual ~CompTransfmEdge() = default;

  virtual void Init() {
    TransfmEdge::Init();
    // struct style
  }
 protected:
  CompTransfmEdge() = default;

 private:
};

class CompTransfmGraph : public TransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(CompTransfmGraph);
  virtual ~CompTransfmGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) override {
    TransformerGraph::Init(task_node, job_has_bp);
    // struct style
  }

  virtual void FwBuildGraph() override {
    FwBuildFromUserOps();
    if (job_has_bp()) {
      FwAddCopyInOp();
    }
    FwAddCloneOp();
    UpdateStartAndStop();
  }

 protected:
  virtual CopyOpConf::CopyType CopyInOpType() = 0;

 private:
  CompTransfmGraph() = default;
  void FwBuildFromUserOps();
  void FwAddCopyInOp();
  void FwAddCloneOp();

  std::unordered_map<std::string, std::vector<CompTransfmNode*>> extern_in_lbn2consumers_;
  std::unordered_map<std::string, std::unique_ptr<BlobDescriptor>> produced_lbn2blob_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
