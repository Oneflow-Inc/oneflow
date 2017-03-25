#ifndef ONEFLOW_GRAPH_BOXING_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_BOXING_EXEC_GRAPH_H_

#include "graph/exec_graph.h"

namespace oneflow {

class BoxingExecGraph final : public ExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingExecGraph);
  BoxingExecGraph() = default;
  ~BoxingExecGraph() = default;

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

#endif // ONEFLOW_GRAPH_BOXING_EXEC_GRAPH_H_
