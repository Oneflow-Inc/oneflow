#ifndef ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/transformer_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CompTransfmGraph : public TransfmGraph {
 public:
  DISALLOW_COPY_AND_MOVE(CompTransfmGraph);
  virtual ~CompTransfmGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) override {
    TransfmGraph::Init(task_node, job_has_bp);
  }

  void FwBuildGraph() override {
    Lbn2NodeMap lbn2producer;
    Lbn2NodeVecMap extern_in_lbn2consumers;
    FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumers);
    if (job_has_bp()) {
      FwAddCopyInOp(&extern_in_lbn2consumers);
    }
    FwAddCloneOp();
    FwSetRelatedTaskEdges(lbn2producer, extern_in_lbn2consumers);
    UpdateStartAndStop();
  }

 protected:
  virtual CopyOpConf::CopyType CopyInOpType() = 0;

 private:
  CompTransfmGraph() = default;

  TransfmEdge* NewTransfmEdge(const std::string& lbn) {
    TransfmEdge* ret = NewFinalEdge();
    ret->mutable_lbn = lbn;
    return ret;
  }

  // Funtions used in FwBuildGraph
  using Lbn2NodeMap = std::unordered_map<std::string, TransfmNode*>;
  using Lbn2NodeVecMap = std::unordered_map<std::string, std::vector<TransfmNode*>>;
  void FwBuildFromUserOps(Lbn2NodeMap* lbn2producer,
                          Lbn2NodeVecMap* extern_in_lbn2consumers);
  void FwAddCopyInOp(Lbn2NodeVecMap* extern_in_lbn2consumers);
  void FwAddCloneOp();
  void FwSetRelatedTaskEdges(const Lbn2NodeMap& lbn2producer,
                             const Lbn2NodeVecMap& extern_in_lbn2consumers);


};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
