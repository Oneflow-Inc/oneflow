#ifndef ONEFLOW_GRAPH_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_OPERATOR_GRAPH_H_

namespace oneflow {

class OpNode : Node {
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

class OpEdge : Edge {
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

class OperatorGraph {
 public:
  DISALLOW_COPY_AND_MOVE(OperatorGraph);
  OperatorGraph() = default;
  virtual ~OperatorGraph() = default;

  virtual void Init(); // TODO

  virtual void BuildOperatorGraph(const TaskNode*) = 0;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_OPERATOR_GRAPH_H_
