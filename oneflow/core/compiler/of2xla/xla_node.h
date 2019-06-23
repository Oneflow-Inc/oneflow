#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_

#include <list>
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace mola {

class XlaNode;
class XlaGraph;

class XlaEdge {
 public:
  bool IsControlEdge() const { return control_edge_; }
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

  XlaEdge() = default;
  XlaEdge(XlaNode *start, XlaNode *end) : start_(start), end_(end) {}
  virtual ~XlaEdge() {}

  // start node of the edge
  XlaNode *start_;
  // end node of the edge
  XlaNode *end_;
  //
  Argument arg_;
  int64_t unique_id_ = -1;
  bool control_edge_ = false;
};

// XLA Node
class XlaNode {
 public:
  const std::list<XlaEdge *> &in_edges() const { return in_edges_; }
  const std::list<XlaEdge *> &out_edges() const { return out_edges_; }
  std::list<XlaEdge *> &in_edges() { return in_edges_; }
  std::list<XlaEdge *> &out_edges() { return out_edges_; }

  void AddInEdge(const XlaEdge *edge);
  void AddOutEdge(const XlaEdge *edge);
  void EraseInEdge(const XlaEdge *edge);
  void EraseOutEdge(const XlaEdge *edge);

  const OpNode *node() const { return node_; }
  const Operator *op() const { return &node_->op(); }
  int64_t unique_id() const { return unique_id_; }
  int64_t cluster_id() const {return cluster_id_; }
  const std::string &backend() const { return backend_; }
  const std::string &op_type() const { return op_type_; }
  const std::string &op_name() const { return op_name_; }

  XlaGraph *sub_graph() const { return sub_graph_; }

  const PbMessage &proto_conf() const { return this->op()->GetCustomizedConf(); }

  void set_cluster_id(int64_t cluster_id) { cluster_id_ = cluster_id; }
  void set_backend(const std::string &backend) { backend_ = backend; }
  void set_op_type(const std::string &type) { op_type_ = type; }
  void set_op_name(const std::string &name) const { op_name_ = name; }

  bool IsCompiled() const { return compiled_; }
  bool IsSourceNode() const;
  bool IsFinishNode() const;

 private:
  friend class XlaGraph;
  // XlaNode only can be created by XlaGraph
  XlaNode() : node_(nullptr), unique_id_(-1), cluster_id_(-1),
              compiled_(false), sub_graph_(nullptr) {}
  explicit XlaNode(const OpNode *op_node);
  virtual ~XlaNode() {}

  // Input edges
  std::list<XlaEdge *> in_edges_;
  // Output edges
  std::list<XlaEdge *> out_edges_;
  // The internal op node which is not holding by this xla node
  const OpNode *node_;
  // Each node has a unique id related to it's index in the graph's nodes
  int64_t unique_id_;
  // Each compiled node has a cluster id if the cluster it belongs is valid.
  // A valid cluster must contain more than `minimum_nodes_in_cluster` nodes
  int64_t cluster_id_;
  // Whether the node can be compiled or not. If the node operator backend and
  // type has been registered by an operator compiler, then `compiled_` is true
  bool compiled_;
  // String device type, such as "CPU" or "CUDA"
  std::string backend_;
  // String operator type, such as "Conv2d", "Matmul" or other else
  std::string op_type_;
  // String operator name
  mutable std::string op_name_;
  // Subgraph will be built for xla launch nodes. Note that `sub_graph_` should
  // be built and managed by the graph, other than the node
  XlaGraph *sub_graph_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
