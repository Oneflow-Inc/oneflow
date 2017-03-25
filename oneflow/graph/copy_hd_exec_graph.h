#ifndef ONEFLOW_GRAPH_COPY_HD_EXEC_H_
#define ONEFLOW_GRAPH_COPY_HD_EXEC_H_

#include "graph/exec_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CopyHDExecGraph final : public ExecGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDExecGraph);
  CopyHDExecGraph() = default;
  ~CopyHDExecGraph() = default;

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

#endif // ONEFLOW_GRAPH_COPY_HD_EXEC_H_
