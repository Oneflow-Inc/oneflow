#ifndef ONEFLOW_GRAPH_COMM_NET_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_COMM_NET_TRANSFM_GRAPH_H_

#include "graph/transfm_graph.h"

namespace oneflow {

class CommNetTransfmGraph final : public TransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetTransfmGraph);
  CommNetTransfmGraph() = default;
  ~CommNetTransfmGraph() = default;

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

#endif // ONEFLOW_GRAPH_COMM_NET_TRANSFM_GRAPH_H_
