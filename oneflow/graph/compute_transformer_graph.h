#ifndef ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/transformer_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CompTransfmGraph : public TransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(CompTransfmGraph);
  virtual ~CompTransfmGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) override {
    TransformerGraph::Init(task_node, job_has_bp);
  }

  void FwBuildGraph() override {
    std::unordered_map<std::string, TransfmNode*> lbn2producer;
    std::unordered_map<std::string, std::vector<TransfmNode*>> extern_in_lbn2consumers;
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

  // Funtions used in FwBuildGraph
  void FwBuildFromUserOps();
  void FwAddCopyInOp();
  void FwAddCloneOp();

  Node dangling_in_edge_src_;
  Node dangling_out_edge_dst_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
