#ifndef ONEFLOW_CORE_GRAPH_REDUCE_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_GRAPH_H_

#include "oneflow/core/graph/graph.h"

namespace oneflow {

class ReduceEdge;
class LogicalNode;
class LogicalGraph;

class ReduceNode final : public Node<ReduceNode, ReduceEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceNode);
  ReduceNode() = default;
  ~ReduceNode() override = default;

  const std::vector<const LogicalNode *> &logical_nodes() const { return logical_nodes_; }
  std::vector<const LogicalNode *> &mut_logical_nodes() { return logical_nodes_; }

 private:
  std::vector<const LogicalNode *> logical_nodes_;
};

class ReduceEdge final : public Edge<ReduceNode, ReduceEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceEdge);
  ReduceEdge() = default;
  ~ReduceEdge() override = default;
};

class ReduceGraph final : public Graph<ReduceNode, ReduceEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGraph);
  explicit ReduceGraph(const LogicalGraph &logical_graph);
  ~ReduceGraph() override = default;

 private:
  struct Group;
  void InitGroups(const LogicalGraph &logical_graph, std::list<Group> *group_list,
                  HashMap<const LogicalNode *, std::list<Group>::iterator> *logical2group_it,
                  HashMap<const LogicalNode *, size_t> *logical2order_in_topo);
  void MergeGroups(std::list<Group> *group_list,
                   HashMap<const LogicalNode *, std::list<Group>::iterator> *logical2group_it);
  bool TryMergeOneGroup(std::list<Group> *group_list,
                        HashMap<const LogicalNode *, std::list<Group>::iterator> *logical2group_it);
  void SortNodesInGroups(std::list<Group> *group_list,
                         HashMap<const LogicalNode *, size_t> *logical2order_in_topo);
  void BuildGraph(const LogicalGraph &logical_graph, std::list<Group> *group_list);
  bool IsLogicalNodeMergeable(const LogicalNode *logical_node) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_GRAPH_H_
