#ifndef ONEFLOW_XRT_GRAPH_NODE_H_
#define ONEFLOW_XRT_GRAPH_NODE_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/xrt/any.h"
#include "oneflow/xrt/graph/algorithm.h"
#include "oneflow/xrt/graph/argument.h"
#include "oneflow/xrt/utility/attribute_map.h"
#include "oneflow/xrt/utility/stl.h"
#include "oneflow/xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

class XrtNode;
class XrtGraph;

class XrtEdge : public util::AttributeMap {
 public:
  XrtNode *start() const { return start_; }
  XrtNode *end() const { return end_; }
  const XrtArgument &argument() const { return arg_; }
  XrtArgument &argument() { return arg_; }

  void SetStartNode(const XrtNode *start) {
    start_ = const_cast<XrtNode *>(start);
  }
  void SetEndNode(const XrtNode *end) { end_ = const_cast<XrtNode *>(end); }
  void SetArgument(const XrtArgument &arg) { arg_ = arg; }

  int64_t unique_id() const { return unique_id_; }
  bool IsControlEdge() const { return !arg_.initialized(); }

 private:
  friend class XrtGraph;

  XrtEdge() = default;
  XrtEdge(const XrtNode *start, const XrtNode *end)
      : start_(const_cast<XrtNode *>(start)),
        end_(const_cast<XrtNode *>(end)) {}

  virtual ~XrtEdge() {}

  XrtNode *start_;
  XrtNode *end_;
  //
  XrtArgument arg_;

  int64_t unique_id_ = -1;
};

// XLA Node
class XrtNode : public util::AttributeMap {
 public:
  const util::List<XrtEdge *> &in_edges() const { return in_edges_; }
  const util::List<XrtEdge *> &out_edges() const { return out_edges_; }
  util::List<XrtEdge *> &in_edges() { return in_edges_; }
  util::List<XrtEdge *> &out_edges() { return out_edges_; }

  void AddInEdge(const XrtEdge *edge);
  void AddOutEdge(const XrtEdge *edge);
  void EraseInEdge(const XrtEdge *edge);
  void EraseOutEdge(const XrtEdge *edge);
  void ClearInEdges() { in_edges_.clear(); };
  void ClearOutEdges() { out_edges_.clear(); };

  int64_t unique_id() const { return unique_id_; }
  const XrtDevice &backend() const { return backend_; }
  const std::string &type() const { return type_; }
  const std::string &name() const { return name_; }
  const PbMessage &param() const { return *param_; }

  XrtGraph *sub_graph() const { return sub_graph_; }

  void set_backend(const XrtDevice &backend) { backend_ = backend; }
  void set_type(const std::string &type) { type_ = type; }
  void set_name(const std::string &name) { name_ = name; }

  bool IsCompiled(const XrtEngine &engine = XrtEngine::XLA) const;
  bool IsSourceNode() const;
  bool IsFinishNode() const;
  bool IsArgumentNode() const;
  bool IsInArgumentNode() const;
  bool IsOutArgumentNode() const;
  bool IsReachable(const XrtNode &dst_node) const;

 protected:
  friend class XrtGraph;

  XrtNode() = default;
  // XrtNode only can be created by XrtGraph
  explicit XrtNode(const PbMessage &param)
      : param_(&param), unique_id_(-1), sub_graph_(nullptr) {}
  virtual ~XrtNode() {}

  util::List<XrtEdge *> in_edges_;
  util::List<XrtEdge *> out_edges_;

  const PbMessage *param_ = nullptr;
  // Each node has a unique id related to it's index in the graph's nodes
  int64_t unique_id_ = -1;
  // Backend device such as X86, CUDA, ARM and so on
  XrtDevice backend_;
  // String type, such as "Conv2d", "Matmul" or other else
  std::string type_;
  // String name
  std::string name_;
  // Subgraph will be built for xrt launch nodes. Note that `sub_graph_` should
  // be built and managed by the graph, other than the node
  XrtGraph *sub_graph_ = nullptr;
};

namespace algorithm {
template <>
struct NodeTypeTrait<XrtNode> {
  typedef XrtEdge *pEdgeType;
};

template <>
struct NodeTypeTrait<const XrtNode> {
  typedef const XrtEdge *pEdgeType;
};
}  // namespace algorithm

bool IsNodeInput(const XrtNode *node, const XrtArgument &argument);
bool IsNodeOutput(const XrtNode *node, const XrtArgument &argument);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_NODE_H_
