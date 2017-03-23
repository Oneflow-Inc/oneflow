#ifndef ONEFLOW_GRAPH_HOST_COMP_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMP_TRANSFM_GRAPH_H_

#include "graph/comp_transfm_graph.h"

namespace oneflow {

class HostCompTransfmGraph final : public CompTransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTransfmGraph);
  HostCompTransfmGraph() = default;
  ~HostCompTransfmGraph() = default;

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::H2H;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMP_TRANSFM_GRAPH_H_
