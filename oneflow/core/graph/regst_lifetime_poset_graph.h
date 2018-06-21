#ifndef ONEFLOW_CORE_GRAPH_REGST_LIFETIME_POSET_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_REGST_LIFETIME_POSET_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class RegstLifetimePosetNode;

class RegstLifetimePosetEdge final : public Edge<RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetEdge);
  RegstLifetimePosetEdge() = default;
  ~RegstLifetimePosetEdge() = default;
};

class RegstLifetimePosetNode final : public Node<RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetNode);
  RegstLifetimePosetNode(const RegstDescProto* regst_desc,
                         std::unique_ptr<HashSet<int64_t>>&& lifetime_actor_ids)
      : regst_desc_(regst_desc), lifetime_actor_ids_(std::move(lifetime_actor_ids)) {}
  ~RegstLifetimePosetNode() = default;

  const RegstDescProto& regst_desc() const { return *regst_desc_; }
  const HashSet<int64_t>& lifetime_actor_ids() const { return *lifetime_actor_ids_; }

 private:
  const RegstDescProto* regst_desc_;
  std::unique_ptr<HashSet<int64_t>> lifetime_actor_ids_;
};

class RegstLifetimePosetGraph final
    : public Graph<const RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetGraph);
  RegstLifetimePosetGraph(
      const std::list<const RegstDescProto*>& regst_descs,
      const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds);
  ~RegstLifetimePosetGraph() = default;

  void ForEachLayerwiseSameColoredRegstDescs(
      const std::function<void(const std::list<const RegstDescProto*>&)>&) const;

 private:
  void InitNodesAndEdges(
      const std::list<const RegstDescProto*>& regst_descs,
      const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds);
  void InitRegstLifetimePosetNode2IntersectedNodes();
  bool LifetimeContain(const RegstLifetimePosetNode* long_lifetime_node,
                       const RegstLifetimePosetNode* short_lifetime_node) const;
  void ForEachSameColoredRegstDescs(
      const HashSet<const RegstLifetimePosetNode*>& layer_nodes,
      const std::function<void(const std::list<const RegstDescProto*>&)>& Handler) const;

  HashMap<const RegstLifetimePosetNode*, HashSet<const RegstLifetimePosetNode*>>
      regst_lifetime_node2intersected_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REGST_LIFETIME_POSET_GRAPH_H_
