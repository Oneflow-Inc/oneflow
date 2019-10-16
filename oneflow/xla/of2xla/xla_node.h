#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_

#include <list>
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/xla/of2xla/xla_argument.h"

namespace oneflow {
namespace mola {

std::string DeviceTypeToBackend(DeviceType device_type);
DeviceType BackendToDeviceType(const std::string &backend);

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

  const SbpParallel &sbp_policy(int index) const;
  const Shape &time_shape(int index) const;
  void set_sbp_policy(int index, const SbpParallel &policy);
  void set_time_shape(int index, const Shape &shape);

 private:
  friend class XlaGraph;

  XlaEdge() = default;
  XlaEdge(XlaNode *start, XlaNode *end) : start_(start), end_(end) {}
  virtual ~XlaEdge() {}

  XlaNode *start_;
  XlaNode *end_;
  //
  Argument arg_;

  SbpParallel sbp_policy_[2];
  Shape time_shape_[2];

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
  void ClearInEdges() { in_edges_.clear(); };
  void ClearOutEdges() { out_edges_.clear(); };

  const Operator *op() const { return op_; }
  int64_t unique_id() const { return unique_id_; }
  int64_t cluster_id() const { return cluster_id_; }
  const std::string &backend() const { return backend_; }
  const std::string &op_type() const { return op_type_; }
  const std::string &op_name() const { return op_name_; }

  std::vector<std::string> input_bns() const;
  std::vector<std::string> output_bns() const;
  const LogicalBlobId &Input(const std::string &bn) const {
    return inputs_.at(bn);
  }
  const LogicalBlobId &Output(const std::string &bn) const {
    return outputs_.at(bn);
  }

  XlaGraph *sub_graph() const { return sub_graph_; }

  virtual const PbMessage &proto_conf() const {
    return this->op()->GetCustomizedConf();
  }

  void set_cluster_id(int64_t cluster_id) { cluster_id_ = cluster_id; }
  void set_backend(const std::string &backend) { backend_ = backend; }
  void set_op_type(const std::string &type) { op_type_ = type; }
  void set_op_name(const std::string &name) { op_name_ = name; }

  bool IsCompiled() const;
  bool IsSourceNode() const;
  bool IsFinishNode() const;
  bool IsArgumentNode() const;
  bool IsInArgumentNode() const;
  bool IsOutArgumentNode() const;
  bool IsReachable(const XlaNode &dst_node) const;

  typedef std::function<BlobDesc *(const LogicalBlobId &)> GetBlobDescFunc;
  virtual void InferBlobDescs(GetBlobDescFunc blob_desc_func,
                              const ParallelContext &parallel_ctx,
                              const SbpSignature &sbp_signature) const;

 protected:
  friend class XlaGraph;
  // XlaNode only can be created by XlaGraph
  XlaNode()
      : op_(nullptr), unique_id_(-1), cluster_id_(-1), sub_graph_(nullptr) {}
  explicit XlaNode(const Operator *op);
  virtual ~XlaNode() {}

  std::list<XlaEdge *> in_edges_;
  std::list<XlaEdge *> out_edges_;
  // The internal op node which is not holded by this xla node
  const Operator *op_;
  // Each node has a unique id related to it's index in the graph's nodes
  int64_t unique_id_;
  // Each compiled node has a cluster id if the cluster it belongs is valid.
  // A valid cluster must contain more than `minimum_nodes_in_cluster` nodes
  int64_t cluster_id_;
  // String device type, such as "CPU" or "CUDA"
  std::string backend_;
  // String operator type, such as "Conv2d", "Matmul" or other else
  std::string op_type_;
  // String operator name
  std::string op_name_;
  // Subgraph will be built for xla launch nodes. Note that `sub_graph_` should
  // be built and managed by the graph, other than the node
  XlaGraph *sub_graph_;
  // Input and output logical blob id
  std::unordered_map<std::string, LogicalBlobId> inputs_;
  std::unordered_map<std::string, LogicalBlobId> outputs_;
};

class XlaArgumentNode : public XlaNode {
 public:
  const PbMessage &proto_conf() const override { return arg_conf_; }

  void InferBlobDescs(XlaNode::GetBlobDescFunc blob_desc_func,
                      const ParallelContext &parallel_ctx,
                      const SbpSignature &sbp_signature) const override;

 private:
  friend class XlaGraph;
  XlaArgumentNode() = default;
  virtual ~XlaArgumentNode() = default;

  explicit XlaArgumentNode(const XlaLaunchOpConf::Argument &arg_conf);

  XlaLaunchOpConf::Argument arg_conf_;
};

bool IsNodeInput(const XlaNode *node, const LogicalBlobId &lbi);
bool IsNodeOutput(const XlaNode *node, const LogicalBlobId &lbi);
bool IsMutableArgument(const XlaNode *node, const Argument &argument);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_NODE_H_
