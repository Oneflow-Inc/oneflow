#ifndef ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_
#define ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_

#include "graph/transformer_graph.h"

namespace oneflow {

class CopyHDTransfmNode final : public TransfmNode {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDTransfmNode);
  CopyHDTransfmNode() = default;
  ~CopyHDTransfmNode() = default;

  void Init() override {
    TransfmNode::Init();
  }

 private:
};

class CopyHDTransfmEdge final : public TransfmEdge {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDTransfmEdge);
  CopyHDTransfmEdge() = default;
  ~CopyHDTransfmEdge() = default;

  void Init() override {
    TransfmEdge::Init();
  }

 private:
};

class CopyHDTransfmGraph final : public TransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(CopyHDTransfmGraph);
  CopyHDTransfmGraph() = default;
  ~CopyHDTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    TransformerGraph::Init(task_node, job_has_bp);
  }

  void FwBuildGraph() override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COPY_HD_TRANSFORMER_H_
