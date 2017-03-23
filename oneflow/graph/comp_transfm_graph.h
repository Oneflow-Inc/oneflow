#ifndef ONEFLOW_GRAPH_COMP_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_COMP_TRANSFM_GRAPH_H_

#include "graph/transfm_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CompTransfmGraph : public TransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTransfmGraph);
  virtual ~CompTransfmGraph() = default;

  void SetupProducedRegisterDesc() override;

 protected:
  virtual CopyOpConf::CopyType CopyInOpType() = 0;
  CompTransfmGraph() = default;

 private:
  void FwBuildGraph() override;
  void BpBuildGraph() override;

  TransfmEdge* NewTransfmEdge(const std::string& lbn) {
    TransfmEdge* ret = NewFinalEdge();
    ret->mut_lbn() = lbn;
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

  // Produced RegisterDesc
  void FwSetupProducedRegisterDesc();
  void BpSetupProducedRegisterDesc();

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMP_TRANSFM_GRAPH_H_
