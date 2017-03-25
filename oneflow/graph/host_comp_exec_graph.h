#ifndef ONEFLOW_GRAPH_HOST_COMP_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMP_EXEC_GRAPH_H_

#include "graph/comp_exec_graph.h"

namespace oneflow {

class HostCompExecGraph final : public CompExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompExecGraph);
  HostCompExecGraph() = default;
  ~HostCompExecGraph() = default;

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::H2H;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMP_EXEC_GRAPH_H_
