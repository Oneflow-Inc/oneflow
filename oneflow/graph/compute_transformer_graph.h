#ifndef ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_

#include "graph/transformer_graph.h"

namespace oneflow {

class ComputeTransformerNode : public TransformerNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeTransformerNode);
  ComputeTransformerNode() = default;
  virtual ~ComputeTransformerNode() = default;

  virtual void Init() {
    TransformerNode::Init();
    // struct style
  }

 private:

};

class ComputeTransformerEdge : public TransformerEdge {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeTransformerEdge);
  ComputeTransformerEdge() = default;
  virtual ~ComputeTransformerEdge() = default;

  virtual void Init() {
    TransformerEdge::Init();
    // struct style
  }

 private:
};

class ComputeTransformerGraph : public TransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeTransformerGraph);
  ComputeTransformerGraph() = default;
  virtual ~ComputeTransformerGraph() = default;

  virtual void Init() {
    TransformerGraph::Init();
    // struct style
  }

 protected:

 private:

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_TRANSFORMER_GRAPH_H_
