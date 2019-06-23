#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

XlaGraph::XlaGraph(const OpGraph *op_graph) {
  CHECK_NOTNULL(op_graph);
  
  std::unordered_map<int64_t, XlaNode *> allocated_nodes; 

  op_graph->TopoForEachNode([&](OpNode *n) -> void {
    XlaNode *node = AddNode(n);
    allocated_nodes.emplace(n->node_id(), node);

    for (const OpEdge *e : n->in_edges()) {
      int64_t node_id = e->src_node()->node_id();
      if (allocated_nodes.count(node_id) > 0) {
        // Generate argument
        CHECK_EQ(e->lbis().size(), 1);
        const LogicalBlobId &lbi = e->lbis()[0];
        Argument argument(lbi, n->LogicalBlobDesc4Lbi(lbi));

        XlaNode *start = allocated_nodes[node_id];
        Connect(start, node, argument);
      }
    }

    for (const OpEdge *e : n->out_edges()) {
      int64_t node_id = e->dst_node()->node_id();
      if (allocated_nodes.count(node_id) > 0) {
        // Generate argument
        CHECK_EQ(e->lbis().size(), 1);
        const LogicalBlobId &lbi = e->lbis()[0];
        Argument argument(lbi, n->LogicalBlobDesc4Lbi(lbi));

        XlaNode *end = allocated_nodes[node_id];
        Connect(node, end, argument);
      }
    }
  });
}

XlaEdge *XlaGraph::Connect(XlaNode *start, XlaNode *end) {
  XlaEdge *edge = AddEdge(start, end);
  start->AddOutEdge(edge);
  end->AddInEdge(edge);
  return edge;
}

XlaEdge *XlaGraph::Connect(XlaNode *start, XlaNode *end, const Argument &arg) {
  XlaEdge *edge = Connect(start, end);
  edge->UpdateArgument(arg);
  return edge;
}

void XlaGraph::Disconnect(XlaEdge *edge) {
  edge->start()->EraseOutEdge(edge);
  edge->end()->EraseInEdge(edge);
}

XlaGraph::~XlaGraph() {
  for (auto &node : nodes_) {
    if (NOTNULL(node)) DELETE(node);
  }
  for (auto &edge : edges_) {
    if (NOTNULL(edge)) DELETE(edge);
  }
  for (auto &pair : subgraphs_) {
    DELETE(pair.second);
  }
}

XlaNode *XlaGraph::Node(int64_t node_id) {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

const XlaNode *XlaGraph::Node(int64_t node_id) const {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

XlaNode *XlaGraph::AddNode() {
  XlaNode *node = new XlaNode();
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
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

XlaGraph *XlaGraph::AddSubGraph(int64_t node_id) {
  CHECK_LT(node_id, nodes_.size());
  auto it = subgraphs_.find(node_id);
  if (it != subgraphs_.end()) {
    DELETE(it->second);
    it->second = new XlaGraph;
  } else {
    it = subgraphs_.emplace(node_id, new XlaGraph).first;
  }
  nodes_[node_id]->sub_graph_ = it->second;
  return it->second;
}

}  // namespace mola
}  // namespace oneflow
