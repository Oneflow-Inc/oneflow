#ifndef ONEFLOW_CORE_GRAPH_CHAIN_LOGICAL_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_LOGICAL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/graph/logical_graph.h"

namespace oneflow {

class ChainLogicalEdge;

class ChainLogicalNode final : public Node<ChainLogicalNode, ChainLogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainLogicalNode);
  ChainLogicalNode() = default;
  ~ChainLogicalNode() override = default;

  const std::vector<const LogicalNode*>& logical_nodes() const { return logical_nodes_; }
  std::vector<const LogicalNode*>& mut_logical_nodes() { return logical_nodes_; }

 private:
  std::vector<const LogicalNode*> logical_nodes_;
};

class ChainLogicalEdge final : public Edge<ChainLogicalNode, ChainLogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainLogicalEdge);
  ChainLogicalEdge() = default;
  ~ChainLogicalEdge() override = default;
};

class ChainLogicalGraph final : public Graph<ChainLogicalNode, ChainLogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainLogicalGraph);
  explicit ChainLogicalGraph(const LogicalGraph& logical_graph);
  ~ChainLogicalGraph() override = default;

 private:
  struct Chain;
  void InitChains(const LogicalGraph& logical_graph, std::list<Chain>* chain_list,
                  HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it,
                  HashMap<const LogicalNode*, size_t>* logical2order_in_topo);
  void MergeChains(std::list<Chain>* chain_list,
                   HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it);
  bool TryMergeTwoChains(std::list<Chain>* chain_list,
                         HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it);
  bool TryMergeTwoParallelChains(
      std::list<Chain>* chain_list,
      HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it);
  bool TryMergeTwoConnectedChains(
      std::list<Chain>* chain_list,
      HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it);
  void SortNodesInChains(std::list<Chain>* chain_list,
                         const HashMap<const LogicalNode*, size_t>& logical2order_in_topo);
  void BuildGraph(const LogicalGraph& logical_graph, std::list<Chain>* chain_list);
  bool IsLogicalNodeMergeable(const LogicalNode* logical_node) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_LOGICAL_GRAPH_H_
