#ifndef ONEFLOW_GRAPH_DEVICE_COMP_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_DEVICE_COMP_EXEC_GRAPH_H_

#include "graph/comp_exec_graph.h"

namespace oneflow {

class DeviceCompExecGraph final : public CompExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompExecGraph);
  DeviceCompExecGraph() = default;
  ~DeviceCompExecGraph() = default;

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DEVICE_COMP_EXEC_GRAPH_H_
