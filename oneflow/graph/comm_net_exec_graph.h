#ifndef ONEFLOW_GRAPH_COMM_NET_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_COMM_NET_EXEC_GRAPH_H_

#include "graph/exec_graph.h"

namespace oneflow {

class CommNetExecGraph final : public ExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetExecGraph);
  CommNetExecGraph() = default;
  ~CommNetExecGraph() = default;

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

#endif // ONEFLOW_GRAPH_COMM_NET_EXEC_GRAPH_H_
