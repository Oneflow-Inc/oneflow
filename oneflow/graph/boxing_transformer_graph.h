#ifndef ONEFLOW_GRAPH_BOXING_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_BOXING_TRANSFORMER_GRAPH_H_

#include "graph/transformer_graph.h"

namespace oneflow {

class BoxingTransfmGraph final : public TransfmGraph {
 public:
  DISALLOW_COPY_AND_MOVE(BoxingTransfmGraph);
  BoxingTransfmGraph() = default;
  ~BoxingTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) {
    TransfmGraph::Init(task_node, job_has_bp);
  }

  void FwBuildGraph() override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TRANSFORMER_GRAPH_H_
