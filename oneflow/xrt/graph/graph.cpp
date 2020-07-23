/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/argument.h"

namespace oneflow {
namespace xrt {

XrtEdge *XrtGraph::Connect(const XrtNode *start, const XrtNode *end) {
  XrtEdge *edge = AddEdge(start, end);
  const_cast<XrtNode *>(start)->AddOutEdge(edge);
  const_cast<XrtNode *>(end)->AddInEdge(edge);
  return edge;
}

XrtEdge *XrtGraph::Connect(const XrtNode *start, const XrtNode *end, const Argument &arg) {
  XrtEdge *edge = Connect(start, end);
  edge->SetArgument(arg);
  return edge;
}

void XrtGraph::Disconnect(const XrtEdge *edge) {
  const_cast<XrtNode *>(edge->start())->EraseOutEdge(edge);
  const_cast<XrtNode *>(edge->end())->EraseInEdge(edge);
}

XrtNode *XrtGraph::Node(int64_t node_id) {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_.at(node_id);
}

const XrtNode *XrtGraph::Node(int64_t node_id) const {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_.at(node_id);
}

XrtNode *XrtGraph::AddNode() {
  std::unique_ptr<XrtNode> node(new XrtNode);
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node.get());
  allocated_nodes_.push_back(std::move(node));
  return nodes_.back();
}

XrtNode *XrtGraph::AddNode(const google::protobuf::Message &param) {
  std::unique_ptr<XrtNode> node(new XrtNode(param));
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node.get());
  allocated_nodes_.push_back(std::move(node));
  return nodes_.back();
}

XrtEdge *XrtGraph::AddEdge() {
  std::unique_ptr<XrtEdge> edge(new XrtEdge);
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge.get());
  allocated_edges_.push_back(std::move(edge));
  return edges_.back();
}

XrtEdge *XrtGraph::AddEdge(const XrtNode *start, const XrtNode *end) {
  std::unique_ptr<XrtEdge> edge(new XrtEdge(start, end));
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge.get());
  allocated_edges_.push_back(std::move(edge));
  return edges_.back();
}

XrtGraph *XrtGraph::AddSubgraph(int64_t node_id) {
  std::unique_ptr<XrtGraph> subgraph(new XrtGraph);
  nodes_[node_id]->sub_graph_ = subgraph.get();
  subgraphs_[node_id] = std::move(subgraph);
  return nodes_.at(node_id)->sub_graph_;
}

std::vector<Argument> XrtGraph::Arguments() const {
  std::vector<Argument> arguments;
  for (const XrtEdge *edge : edges_) {
    if (edge && edge->argument().initialized()) { arguments.push_back(edge->argument()); }
  }
  return std::move(arguments);
}

std::string XrtGraph::ToDot() const {
  std::stringstream ost;
  ost << "digraph {\n";
  for (const XrtNode *node : this->Nodes()) {
    ost << "\"" << node->unique_id() << "\" [label=\"" << node->name() << "\"]\n";
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
