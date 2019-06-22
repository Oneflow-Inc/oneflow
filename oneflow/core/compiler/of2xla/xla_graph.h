#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_

#include <vector>
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"

namespace oneflow {
namespace mola {

class XlaGraph;

class XlaEdge {
 public:
  XlaEdge() = default;
  XlaEdge(XlaNode *start, XlaNode *end) : start_(start), end_(end) {}

  bool IsControlEdge() const;
  XlaNode *start() const { return start_; }
  XlaNode *end() const { return end_; }
  int64_t unique_id() const { return unique_id_; }
  const Argument &argument() const { return arg_; }
  Argument &argument() { return arg_; }

  void UpdateStartNode(XlaNode *start) { start_ = start; }
  void UpdateEndNode(XlaNode *end) { end_ = end; }
  void UpdateArgument(const Argument &arg) { arg_ = arg; }

 private:
  friend class XlaGraph;
  // start node of the edge
  XlaNode *start_;
  // end node of the edge
  XlaNode *end_;
  //  
  Argument arg_;
  int64_t unique_id_ = -1;
};

class XlaGraph {
 public:
  XlaGraph() = default;
  explicit XlaGraph(const OpGraph *op_graph);

  virtual ~XlaGraph();

  XlaNode *Node(int64_t node_id);
  const XlaNode *Node(int64_t node_id) const;

  XlaNode *AddNode(const OpNode *op_node);
  XlaEdge *AddEdge(XlaNode *start, XlaNode *end);

  const std::vector<XlaNode *> &Nodes() const { return nodes_; }
  std::vector<XlaNode *> &Nodes() { return nodes_; }

  void Connect(XlaNode *start, XlaNode *end);
  void Disconnect(XlaNode *start, XlaNode *end);

 private:
  // All allocated nodes in the graph, the node unique id is
  // related to it's index in the vector
  std::vector<XlaNode *> nodes_;
  // All allocated edges in the graph
  std::vector<XlaEdge *> edges_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_NODE_H_
