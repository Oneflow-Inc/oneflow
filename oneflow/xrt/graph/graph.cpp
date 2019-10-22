#include "oneflow/xrt/graph/graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/xrt/graph/argument.h"

namespace oneflow {
namespace xrt {

XrtEdge *XrtGraph::Connect(const XrtNode *start, const XrtNode *end) {
  XrtEdge *edge = AddEdge(start, end);
  const_cast<XrtNode *>(start)->AddOutEdge(edge);
  const_cast<XrtNode *>(end)->AddInEdge(edge);
  return edge;
}

XrtEdge *XrtGraph::Connect(const XrtNode *start, const XrtNode *end,
                           const XrtArgument &arg) {
  XrtEdge *edge = Connect(start, end);
  edge->SetArgument(arg);
  return edge;
}

void XrtGraph::Disconnect(const XrtEdge *edge) {
  const_cast<XrtNode *>(edge->start())->EraseOutEdge(edge);
  const_cast<XrtNode *>(edge->end())->EraseInEdge(edge);
}

XrtGraph::~XrtGraph() {
  DeleteNodes();
  DeleteEdges();
  DeleteSubgraphs();
}

void XrtGraph::DeleteNodes() {
  for (auto &node : nodes_) {
    if (node) delete node;
  }
}

void XrtGraph::DeleteEdges() {
  for (auto &edge : edges_) {
    if (edge) delete edge;
  }
}

void XrtGraph::DeleteSubgraphs() {
  for (auto &graph : subgraphs_) {
    if (graph.second) delete graph.second;
  }
}

XrtNode *XrtGraph::Node(int64_t node_id) {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

const XrtNode *XrtGraph::Node(int64_t node_id) const {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

XrtNode *XrtGraph::AddNode() {
  XrtNode *node = new XrtNode();
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XrtNode *XrtGraph::AddNode(const PbMessage &param) {
  XrtNode *node = new XrtNode(param);
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XrtEdge *XrtGraph::AddEdge() {
  XrtEdge *edge = new XrtEdge();
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge);
  return edge;
}

XrtEdge *XrtGraph::AddEdge(const XrtNode *start, const XrtNode *end) {
  XrtEdge *edge = new XrtEdge(start, end);
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge);
  return edge;
}

XrtGraph *XrtGraph::AddSubgraph(int64_t node_id) {
  CHECK_LT(node_id, nodes_.size());
  auto it = subgraphs_.find(node_id);
  if (it != subgraphs_.end()) {
    delete it->second;
    it->second = new XrtGraph;
  } else {
    it = subgraphs_.emplace(node_id, new XrtGraph).first;
  }
  nodes_[node_id]->sub_graph_ = it->second;
  return it->second;
}

std::vector<XrtArgument> XrtGraph::Arguments() const {
  std::vector<XrtArgument> arguments;
  for (const XrtEdge *edge : edges_) {
    if (edge && edge->argument().is_initialized()) {
      arguments.push_back(edge->argument());
    }
  }
  return std::move(arguments);
}

std::string XrtGraph::ToDot() const {
  std::stringstream ost;
  ost << "digraph {\n";
  for (const XrtNode *node : this->Nodes()) {
    ost << "\"" << node->unique_id() << "\" [label=\"" << node->name()
        << "\"]\n";
  }
  for (const XrtEdge *edge : edges_) {
    ost << "\"" << edge->start()->unique_id() << "\" -> "
        << "\"" << edge->end()->unique_id() << "\"\n";
  }
  ost << "}";
  return ost.str();
}

}  // namespace xrt
}  // namespace oneflow
