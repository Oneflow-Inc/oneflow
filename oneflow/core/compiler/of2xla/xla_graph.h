#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_

#include <vector>
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"

namespace oneflow {
namespace mola {

class XlaGraph {
 public:
  XlaGraph() = default;
  explicit XlaGraph(const OpGraph *op_graph);

  virtual ~XlaGraph();

  XlaNode *Node(int64_t node_id);
  const XlaNode *Node(int64_t node_id) const;

  XlaNode *AddNode();
  XlaNode *AddNode(const OpNode *op_node);
  XlaEdge *AddEdge(XlaNode *start, XlaNode *end);

  // Create a subgraph for node that unique id is `node_id`
  XlaGraph *AddSubGraph(int64_t node_id);

  const std::vector<XlaNode *> &Nodes() const { return nodes_; }
  std::vector<XlaNode *> &Nodes() { return nodes_; }

  XlaEdge *Connect(XlaNode *start, XlaNode *end);
  XlaEdge *Connect(XlaNode *start, XlaNode *end, const Argument &arg);
  void Disconnect(XlaEdge *edge);

 private:
  // All allocated nodes in the graph. The node unique id is related to it's
  // index in the vector. The Xla node in `nodes_` can be nullptr since we will
  // always keep it in `nodes_` even if it has been removed from the graph.
  std::vector<XlaNode *> nodes_;

  // All allocated edges in the graph. The edge unique id is related to it's
  // index in the vector. And the xla edge in `edges_` can also be nullptr.
  std::vector<XlaEdge *> edges_;

  // All allocated subgraphs. The key of the map means node unique id, and the
  // value is the subgraph which belongs to the node
  std::unordered_map<int64_t, XlaGraph *> subgraphs_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_NODE_H_
