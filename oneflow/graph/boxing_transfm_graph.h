#ifndef ONEFLOW_GRAPH_BOXING_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_BOXING_TRANSFM_GRAPH_H_

#include "graph/transfm_graph.h"

namespace oneflow {

class BoxingTransfmGraph final : public TransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTransfmGraph);
  BoxingTransfmGraph() = default;
  ~BoxingTransfmGraph() = default;

  void SetupProducedRegisterDesc() override {
    LOG(FATAL) << "TODO";
  }

 private:
  void FwBuildGraph() override {
    LOG(FATAL) << "TODO";
  }
  void BpBuildGraph() override {
    LOG(FATAL) << "TODO";
  }
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TRANSFM_GRAPH_H_
