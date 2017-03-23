#ifndef ONEFLOW_GRAPH_COPY_HD_TRANSFM_H_
#define ONEFLOW_GRAPH_COPY_HD_TRANSFM_H_

#include "graph/transfm_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CopyHDTransfmGraph final : public TransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDTransfmGraph);
  CopyHDTransfmGraph() = default;
  ~CopyHDTransfmGraph() = default;

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

#endif // ONEFLOW_GRAPH_COPY_HD_TRANSFM_H_
