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
#ifndef ONEFLOW_XRT_GRAPH_GRAPH_H_
#define ONEFLOW_XRT_GRAPH_GRAPH_H_

#include <google/protobuf/message.h>
#include <vector>

#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/algorithm.h"
#include "oneflow/xrt/graph/node.h"
#include "oneflow/xrt/utility/attribute_map.h"

namespace oneflow {
namespace xrt {

class XrtGraph : public util::AttributeMap {
 public:
  XrtGraph() = default;
  virtual ~XrtGraph() = default;

  XrtNode *Node(int64_t node_id);
  const XrtNode *Node(int64_t node_id) const;

  XrtNode *AddNode();
  XrtNode *AddNode(const google::protobuf::Message &param);

  XrtEdge *AddEdge();
  XrtEdge *AddEdge(const XrtNode *start, const XrtNode *end);

  XrtEdge *Connect(const XrtNode *start, const XrtNode *end);
  XrtEdge *Connect(const XrtNode *start, const XrtNode *end, const Argument &arg);
  void Disconnect(const XrtEdge *edge);

  // Create a subgraph for node that unique id is `node_id`
  XrtGraph *AddSubgraph(int64_t node_id);

  const std::vector<XrtNode *> &Nodes() const { return nodes_; }
  std::vector<XrtNode *> &Nodes() { return nodes_; }

  const std::vector<XrtEdge *> &Edges() const { return edges_; }
  std::vector<XrtEdge *> &Edges() { return edges_; }

  std::string ToDot() const;

  std::vector<Argument> Arguments() const;

 protected:
  std::vector<XrtNode *> nodes_;
  // All allocated nodes in the graph. The node unique id is related to it's
  // index in the vector. The Xrt node in `nodes_` can be nullptr since we will
  // always keep it in `nodes_` even if it has been removed from the graph.
  std::vector<std::unique_ptr<XrtNode>> allocated_nodes_;

  std::vector<XrtEdge *> edges_;
  // All allocated edges in the graph. The edge unique id is related to it's
  // index in the vector. And the xrt edge in `edges_` can also be nullptr.
  std::vector<std::unique_ptr<XrtEdge>> allocated_edges_;

  // All allocated subgraphs. The key of the map means node unique id, and the
  // value is the subgraph which belongs to the node
  util::Map<int64_t, std::unique_ptr<XrtGraph>> subgraphs_;
};

namespace algorithm {
template<>
struct GraphTypeTrait<XrtGraph> {
  typedef XrtNode *pNodeType;
  typedef XrtEdge *pEdgeType;
};

template<>
struct GraphTypeTrait<const XrtGraph> {
  typedef const XrtNode *pNodeType;
  typedef const XrtEdge *pEdgeType;
};
}  // namespace algorithm

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_GRAPH_H_
