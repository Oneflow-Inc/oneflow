#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_PASS_CLUSTER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_PASS_CLUSTER_H_

#include <unordered_map>
#include <unordered_set>
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_utility.h"

namespace oneflow {
namespace mola {

namespace util {
template <typename K, typename T>
using Map = std::unordered_map<K, T>;
template <typename T>
using Set = std::unordered_set<T>;
}  // namespace util

class ClusterNode;
class ClusterEdge {
 public:
  ClusterEdge() = default;
  ClusterEdge(ClusterNode *start, ClusterNode *end)
      : start_(start), end_(end) {}
  virtual ~ClusterEdge() {}

  void UpdateStartNode(ClusterNode *start) { start_ = start; }
  void UpdateEndNode(ClusterNode *end) { end_ = end; }

  ClusterNode *start() const { return start_; }
  ClusterNode *end() const { return end_; }

  bool is_control_edge() const { return is_control_edge_; }
  void set_is_control_edge(bool is_control_edge) {
    is_control_edge_ = is_control_edge;
  }

  bool is_fusion_disabled() const { return is_fusion_disabled_; }
  void set_is_fusion_disabled(bool is_fusion_disabled) {
    is_fusion_disabled_ = is_fusion_disabled;
  }

  bool IsIdentity() const {
    return is_control_edge_ || (start_sbp_policy() == end_sbp_policy());
  }

  SbpParallel start_sbp_policy() const { return sbp_policy_[0]; }
  SbpParallel end_sbp_policy() const { return sbp_policy_[1]; }
  void set_start_sbp_policy(const SbpParallel &sbp_policy) {
    sbp_policy_[0] = sbp_policy;
  }
  void set_end_sbp_policy(const SbpParallel &sbp_policy) {
    sbp_policy_[1] = sbp_policy;
  }
  Shape start_time_shape() const { return time_shape_[0]; }
  Shape end_time_shape() const { return time_shape_[1]; }
  void set_start_time_shape(const Shape &shape) { time_shape_[0] = shape; }
  void set_end_time_shape(const Shape &shape) { time_shape_[1] = shape; }

 protected:
  ClusterNode *start_;
  ClusterNode *end_;
  SbpParallel sbp_policy_[2];
  Shape time_shape_[2];
  bool is_control_edge_ = false;
  bool is_fusion_disabled_ = false;
};

class ClusterNode {
 public:
  ClusterNode() : ClusterNode(nullptr, -1) {}
  explicit ClusterNode(int64_t id) : ClusterNode(nullptr, id) {}
  explicit ClusterNode(const XlaNode *node, int64_t id)
      : xrt_node_(node), id_(id) {
    folded_nodes_.insert(this);
  }
  virtual ~ClusterNode() {}

  util::Set<ClusterEdge *> &in_edges() { return in_edges_; }
  util::Set<ClusterEdge *> &out_edges() { return out_edges_; }
  const util::Set<ClusterEdge *> &in_edges() const { return in_edges_; }
  const util::Set<ClusterEdge *> &out_edges() const { return out_edges_; }

  void AddInEdge(const ClusterEdge *edge);
  void AddOutEdge(const ClusterEdge *edge);
  void EraseInEdge(const ClusterEdge *edge);
  void EraseOutEdge(const ClusterEdge *edge);
  void ClearInEdges() { in_edges_.clear(); }
  void ClearOutEdges() { out_edges_.clear(); }

  void FoldNodes(const util::Set<ClusterNode *> &nodes) {
    folded_nodes_.insert(nodes.begin(), nodes.end());
  }

  void Merge(ClusterNode &other);
  bool TryMerge(ClusterNode &other);
  bool IsReachable(const ClusterNode &target);
  bool IsSatisfySbpPolicy() const;
  bool IsSourceNode() const { return in_edges_.empty(); }
  virtual bool IsCompiled() const { return xrt_node_->IsCompiled(); }

  bool operator==(const ClusterNode &other) const { return id_ == other.id_; }

  const XlaNode *xrt_node() const { return xrt_node_; }
  int64_t id() const { return id_; }
  void set_id(int64_t id) { id_ = id; }
  virtual std::string type() const { return xrt_node_->op_type(); }
  virtual std::string name() const { return xrt_node_->op_name(); }
  virtual std::string backend() const { return xrt_node_->backend(); }

  size_t size() const { return folded_nodes_.size(); }
  const util::Set<ClusterNode *> &folded_nodes() const { return folded_nodes_; }
  util::Set<ClusterNode *> &folded_nodes() { return folded_nodes_; }

 protected:
  const XlaNode *xrt_node_;
  int64_t id_;
  util::Set<ClusterNode *> folded_nodes_;
  util::Set<ClusterEdge *> in_edges_;
  util::Set<ClusterEdge *> out_edges_;
};

class NoClusterNode : public ClusterNode {
 public:
  NoClusterNode(int64_t id) : ClusterNode(id), backend_("") {
    name_ = absl::StrCat("#node_", InstanceCount());
  }
  virtual ~NoClusterNode() {}
  // Not thread-safe
  int64_t InstanceCount() {
    static int64_t instance_count = 0;
    int64_t count = instance_count++;
    return count;
  }

  void set_backend(const std::string &backend) { backend_ = backend; }

  bool IsCompiled() const override { return true; }
  std::string type() const override { return "NoClusterNode"; }
  std::string name() const override { return name_; }
  std::string backend() const override { return backend_; }

 private:
  std::string name_;
  std::string backend_;
};

bool IsNodeDirectChildren(const ClusterNode *parent,
                          const ClusterNode *children);

bool IsNoClusterNode(const ClusterNode *node);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_PASS_CLUSTER_H_
