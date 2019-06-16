#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

bool XlaEdge::IsControlEdge() const {
  // TODO
  return false;
}

XlaGraph::XlaGraph(const OpGraph *op_graph) {
  CHECK_NOTNULL(op_graph);
  
  std::unordered_map<int64_t, XlaNode *> allocated_nodes; 

  op_graph->TopoForEachNode([&](OpNode *n) -> void {
    XlaNode *node = AddNode(n);
    allocated_nodes.emplace(n->node_id(), node);

    for (const OpEdge *e : n->in_edges()) {
      int64_t node_id = e->src_node()->node_id();
      if (allocated_nodes.count(node_id) > 0) {
        XlaNode *start = allocated_nodes[node_id];
        Connect(start, node);
      }
    }

    for (const OpEdge *e : n->out_edges()) {
      int64_t node_id = e->dst_node()->node_id();
      if (allocated_nodes.count(node_id) > 0) {
        XlaNode *end = allocated_nodes[node_id];
        Connect(node, end);
      }
    }
  });
}

void XlaGraph::Connect(XlaNode *start, XlaNode *end) {
  XlaEdge *edge = AddEdge(start, end);
  start->AddOutEdge(edge);
  end->AddInEdge(edge);
}

XlaGraph::~XlaGraph() {
  for (auto &node : nodes_) {
    if (NOTNULL(node)) delete node;
  }
  for (auto &edge : edges_) {
    if (NOTNULL(edge)) delete edge;
  }
}

void XlaGraph::CopyFrom(const XlaGraph &graph) {

}

XlaNode *XlaGraph::Node(int64_t node_id) {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

const XlaNode *XlaGraph::Node(int64_t node_id) const {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

XlaNode *XlaGraph::AddNode(const OpNode *op_node) {
  XlaNode *node = new XlaNode(op_node);
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XlaEdge *XlaGraph::AddEdge(XlaNode *start, XlaNode *end) {
  XlaEdge *edge = new XlaEdge(start, end);
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge);
  return edge;
}

}  // namespace mola
}  // namespace oneflow
