#ifndef ONEFLOW_GRAPH_COMP_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_COMP_EXEC_GRAPH_H_

#include "graph/exec_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CompExecGraph : public ExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompExecGraph);
  virtual ~CompExecGraph() = default;

  void SetupProducedRegisterDesc() override;

 protected:
  virtual CopyOpConf::CopyType CopyInOpType() = 0;
  CompExecGraph() = default;

 private:
  void FwBuildGraph() override;
  void BpBuildGraph() override;

  ExecEdge* NewExecEdge(const std::string& lbn) {
    ExecEdge* ret = NewFinalEdge();
    ret->mut_lbn() = lbn;
    return ret;
  }

  // Funtions used in FwBuildGraph
  using Lbn2NodeMap = std::unordered_map<std::string, ExecNode*>;
  using Lbn2NodeVecMap = std::unordered_map<std::string, std::vector<ExecNode*>>;
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

#endif // ONEFLOW_GRAPH_COMP_EXEC_GRAPH_H_
