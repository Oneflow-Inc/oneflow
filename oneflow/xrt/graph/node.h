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
#ifndef ONEFLOW_XRT_GRAPH_NODE_H_
#define ONEFLOW_XRT_GRAPH_NODE_H_

#include <google/protobuf/message.h>

#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/algorithm.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/attribute_map.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

class XrtNode;
class XrtGraph;

class XrtEdge : public util::AttributeMap {
 public:
  XrtNode* start() const { return start_; }
  XrtNode* end() const { return end_; }
  const Argument& argument() const { return arg_; }
  Argument& argument() { return arg_; }

  void SetStartNode(const XrtNode* start) { start_ = const_cast<XrtNode*>(start); }
  void SetEndNode(const XrtNode* end) { end_ = const_cast<XrtNode*>(end); }
  void SetArgument(const Argument& arg) { arg_ = arg; }

  int64_t unique_id() const { return unique_id_; }

  bool IsControlEdge() const { return !arg_.initialized(); }

  virtual ~XrtEdge() = default;

  friend class XrtGraph;

 protected:
  XrtEdge() = default;
  XrtEdge(const XrtNode* start, const XrtNode* end)
      : start_(const_cast<XrtNode*>(start)), end_(const_cast<XrtNode*>(end)) {}

 protected:
  XrtNode* start_ = nullptr;
  XrtNode* end_ = nullptr;
  Argument arg_;
  int64_t unique_id_ = -1;
};

// XLA Node
class XrtNode : public util::AttributeMap {
 public:
  const util::List<XrtEdge*>& in_edges() const { return in_edges_; }
  const util::List<XrtEdge*>& out_edges() const { return out_edges_; }
  util::List<XrtEdge*>& in_edges() { return in_edges_; }
  util::List<XrtEdge*>& out_edges() { return out_edges_; }

  void AddInEdge(const XrtEdge* edge);
  void AddOutEdge(const XrtEdge* edge);
  void EraseInEdge(const XrtEdge* edge);
  void EraseOutEdge(const XrtEdge* edge);
  void ClearInEdges() { in_edges_.clear(); };
  void ClearOutEdges() { out_edges_.clear(); };

  int64_t unique_id() const { return unique_id_; }
  const XrtDevice& device() const { return device_; }
  const std::string& type() const { return type_; }
  const std::string& name() const { return name_; }

  const google::protobuf::Message& param() const { return *param_; }

  XrtGraph* sub_graph() const { return sub_graph_; }

  void set_device(const XrtDevice& device) { device_ = device; }
  void set_type(const std::string& type) { type_ = type; }
  void set_name(const std::string& name) { name_ = name; }

  bool IsSourceNode() const;
  bool IsFinishNode() const;
  bool IsArgumentNode() const;
  bool IsInArgumentNode() const;
  bool IsOutArgumentNode() const;

  bool IsReachable(const XrtNode& dst_node) const;

  virtual ~XrtNode() {}

  friend class XrtGraph;

 protected:
  XrtNode() = default;
  // XrtNode only can be created by XrtGraph
  explicit XrtNode(const google::protobuf::Message& param)
      : param_(&param), unique_id_(-1), sub_graph_(nullptr) {}

 protected:
  util::List<XrtEdge*> in_edges_;
  util::List<XrtEdge*> out_edges_;

  const google::protobuf::Message* param_ = nullptr;
  // Each node has a unique id related to it's index in the graph's nodes
  int64_t unique_id_ = -1;
  // Backend device such as X86, CUDA, ARM and so on
  XrtDevice device_;
  // String type, such as "Conv2d", "Matmul" or other else
  std::string type_;
  // String name
  std::string name_;
  // Subgraph will be built for xrt launch nodes. Note that `sub_graph_` should
  // be built and managed by the graph, other than the node
  XrtGraph* sub_graph_ = nullptr;
};

namespace algorithm {
template<>
struct NodeTypeTrait<XrtNode> {
  typedef XrtEdge* pEdgeType;
};

template<>
struct NodeTypeTrait<const XrtNode> {
  typedef const XrtEdge* pEdgeType;
};
}  // namespace algorithm

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_NODE_H_
