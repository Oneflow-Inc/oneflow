#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_

#include <vector>
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace mola {

class XlaEdge;
class XlaGraph;

// XLA Node
class XlaNode {
 public:
  virtual ~XlaNode() {}

  const std::vector<XlaEdge *> &in_edges() const { return in_edges_; }
  const std::vector<XlaEdge *> &out_edges() const { return out_edges_; }
  std::vector<XlaEdge *> &in_edges() { return in_edges_; }
  std::vector<XlaEdge *> &out_edges() { return out_edges_; }
  void AddInEdge(const XlaEdge *edge);
  void AddOutEdge(const XlaEdge *edge);

  const OpNode *node() const { return node_; }
  const Operator *op() const { return &node_->op(); }
  const int64_t unique_id() const { return unique_id_; }
  const int64_t cluster_id() const {return cluster_id_; }
  const std::string &backend() const { return backend_; }
  const std::string &op_type() const { return op_type_; }

  const PbMessage &proto_conf() const { return this->op()->GetCustomizedConf(); }

  void set_cluster_id(int64_t cluster_id) { cluster_id_ = cluster_id; }

  bool IsCompiled() const { return compiled_; }
  bool IsSourceNode() const;
  bool IsFinishNode() const;

 private:
  friend class XlaGraph;
  // XlaNode only can be created by XlaGraph
  XlaNode() : node_(nullptr), unique_id_(-1), compiled_(false) {}
  explicit XlaNode(const OpNode *op_node);

  // Input edges. TODO(hjchen2) use set other than vector
  std::vector<XlaEdge *> in_edges_;
  // Output edges
  std::vector<XlaEdge *> out_edges_;
  // The internal op node which is not holding by this xla node
  const OpNode *node_;
  // Each node has a unique id related to it's index in graph's nodes
  int64_t unique_id_;
  // Each node has a cluster id if the cluster it belongs is valid.
  // A valid cluster must have more than `minimum_nodes_in_cluster` nodes
  int64_t cluster_id_;
  // Whether the node can be compiled or not. If the node operator backend and
  // type has been registered by an operator compiler, then compiled is true.
  bool compiled_;
  // String device type, such as "CPU" or "CUDA"
  std::string backend_;
  // String operator type, such as "Conv2d", "Matmul" or other else
  std::string op_type_;
};

}
}

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
