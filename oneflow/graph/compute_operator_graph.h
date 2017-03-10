#ifndef ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_

#include "graph/operator_graph.h"

namespace oneflow {

class ComputeOpNode : OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOpNode);
  ComputeOpNode() = default;
  virtual ~ComputeOpNode() = default;

  virtual void Init() {
    OpNode::Init();
    // struct style
  }

 private:

};

class ComputeOpEdge : OpEdge {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOpEdge);
  ComputeOpEdge() = default;
  virtual ~ComputeOpEdge() = default;

  virtual void Init() {
    OpEdge::Init();
  }

 private:
};

class ComputeOperatorGraph : public OperatorGraph {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOperatorGraph);
  ComputeOperatorGraph() = default;
  virtual ComputeOperatorGraph() = default;

  virtual void Init() {
    OperatorGraph::Init();
  }

 protected:

 private:

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_
