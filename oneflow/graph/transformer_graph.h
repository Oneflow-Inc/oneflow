#ifndef ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class TransformerNode : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(TransformerNode);
  TransformerNode() = default;
  virtual ~TransformerNode() = default;

  virtual void Init() {
    Node::Init();
    // struct style
  }

 private:
};

class TransformerEdge : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(TransformerEdge);
  TransformerEdge() = default;
  virtual ~TransformerEdge() = default;

  virtual void Init() {
    Edge::Init();
    // struct style
  }

 private:
};

class TransformerGraph : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TransformerGraph);
  TransformerGraph() = default;
  virtual ~TransformerGraph() = default;

  virtual void Init() {
    Graph::Init();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
