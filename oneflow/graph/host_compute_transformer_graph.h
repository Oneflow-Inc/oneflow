#ifndef ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/compute_transformer_graph.h"

namespace oneflow {

class HostCompTransfmGraph final : public CompTransfmGraph {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompTransfmGraph);
  HostCompTransfmGraph() = default;
  ~HostCompTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    ComputeTransfmGraph::Init(task_node, job_has_bp);
  }

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::H2H;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMPUTE_TRANSFORMER_GRAPH_H_
