#ifndef ONEFLOW_GRAPH_STAGE_GRAPH_H_
#define ONEFLOW_GRAPH_STAGE_GRAPH_H_

#include "graph/chain_graph.h"

namespace oneflow {

class StageNode final : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(StageNode);
  StageNode() = default;
  ~StageNode() = default;

  void Init() {
    Node::Init();
    // struct style
  }

  const MachineId& machine_id() const {
    return machine_id_;
  }
  MachineId& mutable_machine_id() {
    return machine_id_;
  }

  const ChainNode* chain_node() const {
    return chain_node_;
  }
  void set_chain_node(const ChainNode* new_chain_node) {
    chain_node_ = new_chain_node;
  }

 private:
  const ChainNode* chain_node_;
  MachineId machine_id_;

};

class StageEdge final : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(StageEdge);
  StageEdge() = default;
  ~StageEdge() = default;
    
  void Init() {
    Edge::Init();
  }
    
 private:
};

class StageGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(StageGraph);
  StageGraph() = default;
  ~StageGraph() = default;

  void Init(std::shared_ptr<const ChainGraph> chain_graph);

 private:
  StageNode* NewStageNode() {
    StageNode* ret_ptr = new StageNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }
  StageEdge* NewStageEdge() {
    StageEdge* ret_ptr = new StageEdge;
    ret_ptr->Init();
    RegisterEdge(ret_ptr);
    return ret_ptr;
  }
  std::shared_ptr<const ChainGraph> chain_graph_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_STAGE_GRAPH_H_
