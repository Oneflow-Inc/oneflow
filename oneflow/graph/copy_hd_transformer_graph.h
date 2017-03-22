#ifndef ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_
#define ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_

#include "graph/transformer_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class CopyHDTransfmGraph final : public TransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHDTransfmGraph);
  CopyHDTransfmGraph() = default;
  ~CopyHDTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    TransfmGraph::Init(task_node, job_has_bp);
  }

  void FwBuildGraph() override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_
