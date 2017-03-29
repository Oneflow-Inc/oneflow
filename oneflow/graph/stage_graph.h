#ifndef ONEFLOW_GRAPH_STAGE_GRAPH_H_
#define ONEFLOW_GRAPH_STAGE_GRAPH_H_

#include "common/range.h"
#include "graph/chain_graph.h"

namespace oneflow {

class StageEdge;

class StageNode final : public Node<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageNode);
  StageNode() = default;
  ~StageNode() = default;

  const MachineId& machine_id() const {
    return machine_id_;
  }
  MachineId& mut_machine_id() {
    return machine_id_;
  }

  const ChainNode* chain_node() const {
    return chain_node_;
  }
  void set_chain_node(const ChainNode* new_chain_node) {
    chain_node_ = new_chain_node;
  }

  const Range& parallel_range() const {
    return parallel_range_;
  }
  Range& mut_parallel_range() {
    return parallel_range_;
  }

 private:
  const ChainNode* chain_node_;
  MachineId machine_id_;
  Range parallel_range_;

};

class StageEdge final : public Edge<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageEdge);
  StageEdge() = default;
  ~StageEdge() = default;
    
 private:
};

class StageGraph final : public Graph<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageGraph);
  StageGraph() = default;
  ~StageGraph() = default;

  void Init(std::unique_ptr<const ChainGraph>&& chain_graph);

  const ChainGraph* chain_graph() const { return chain_graph_.get(); }

 private:
  std::unique_ptr<const ChainGraph> chain_graph_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_STAGE_GRAPH_H_
