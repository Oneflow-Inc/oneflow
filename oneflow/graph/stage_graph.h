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

  std::shared_ptr<std::vector<std::shared_ptr<const Operator>>> op_vec_ptr() const {
    return op_vec_ptr_;
  }
  std::shared_ptr<std::vector<std::shared_ptr<const Operator>>>& mutable_op_vec_ptr() {
    return op_vec_ptr_;
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

  const MachineId& machine_id() const {
    return machine_id_;
  }
  MachineId& mutable_machine_id() {
    return machine_id_;
  }

  std::shared_ptr<const std::vector<std::string>> input_lbns_ptr() const {
    return input_lbns_ptr_;
  }
  std::shared_ptr<const std::vector<std::string>>& mutable_input_lbns_ptr() {
    return input_lbns_ptr_;
  }
  
  std::shared_ptr<std::vector<std::string>> output_lbns_ptr() const {
    return output_lbns_ptr_;
  }
  std::shared_ptr<std::vector<std::string>>& mutable_output_lbns_ptr() {
    return output_lbns_ptr_;
  }

 private:
  std::shared_ptr<const std::vector<std::shared_ptr<const Operator>>> op_vec_ptr_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;
  MachineId machine_id_;
  std::shared_ptr<const std::vector<std::string>> input_lbns_ptr_;
  std::shared_ptr<const std::vector<std::string>> output_lbns_ptr_;

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

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_STAGE_GRAPH_H_
