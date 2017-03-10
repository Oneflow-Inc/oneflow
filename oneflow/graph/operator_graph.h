#ifndef ONEFLOW_GRAPH_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_OPERATOR_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class OpNode : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(OpNode);
  OpNode() = default;
  virtual ~OpNode() = default;

  virtual void Init() {
    Node::Init();
    // struct style
  }

 private:
};

class OpEdge : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(OpEdge);
  OpEdge() = default;
  virtual ~OpEdge() = default;

  virtual void Init() {
    Edge::Init();
    // struct style
  }

 private:
};

class OperatorGraph : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(OperatorGraph);
  OperatorGraph() = default;
  virtual ~OperatorGraph() = default;

  virtual void Init() {
    Graph::Init();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_OPERATOR_GRAPH_H_
