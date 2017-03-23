#ifndef ONEFLOW_GRAPH_DEVICE_COMP_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_DEVICE_COMP_TRANSFM_GRAPH_H_

#include "graph/comp_transfm_graph.h"

namespace oneflow {

class DeviceCompTransfmGraph final : public CompTransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTransfmGraph);
  DeviceCompTransfmGraph() = default;
  ~DeviceCompTransfmGraph() = default;

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DEVICE_COMP_TRANSFM_GRAPH_H_
