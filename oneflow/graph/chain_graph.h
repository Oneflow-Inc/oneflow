#ifndef ONEFLOW_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_GRAPH_CHAIN_GRAPH_H_

#include <list>
#include "graph/logical_graph.h"

namespace oneflow {

class ChainNode final : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(ChainNode);
  ChainNode() = default;
  ~ChainNode() = default;

  void Init() {
    Node::Init();
    // struct style
  }

  const std::vector<std::shared_ptr<const Operator>>& op_vec() const {
    return op_vec_;
  }
  std::vector<std::shared_ptr<const Operator>>& mutable_op_vec() {
    return op_vec_;
  }

  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class ChainEdge final : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;
    
  void Init() {
    Edge::Init();
  }

 private:
};

class ChainGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = default;
  ~ChainGraph() = default;

  void Init(const LogicalGraph* logical_graph);

 private:
  ChainNode* NewChainNode() {
    ChainNode* ret_ptr = new ChainNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }
  ChainEdge* NewChainEdge() {
    ChainEdge* ret_ptr = new ChainEdge;
    ret_ptr->Init();
    RegisterEdge(ret_ptr);
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_CHAIN_GRAPH_H_
