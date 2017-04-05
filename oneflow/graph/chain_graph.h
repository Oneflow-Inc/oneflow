#ifndef ONEFLOW_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_GRAPH_CHAIN_GRAPH_H_

#include <list>
#include "graph/logical_graph.h"

namespace oneflow {

class ChainEdge;

class ChainNode final : public Node<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  ChainNode() = default;
  ~ChainNode() = default;

  std::string ConcatedOpsName() const;

  const std::vector<std::shared_ptr<Operator>>& op_vec() const {
    return op_vec_;
  }
  std::vector<std::shared_ptr<Operator>>& mut_op_vec() {
    return op_vec_;
  }

  std::shared_ptr<const ParallelDesc> parallel_desc() const {
    return parallel_desc_;
  }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() {
    return parallel_desc_;
  }

  const std::vector<std::string>& input_lbns() const {
    return input_lbns_;
  }
  std::vector<std::string>& mut_input_lbns() {
    return input_lbns_;
  }
  
  const std::vector<std::string>& output_lbns() const {
    return output_lbns_;
  }
  std::vector<std::string>& mut_output_lbns() {
    return output_lbns_;
  }

 private:
  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
  std::vector<std::string> input_lbns_;
  std::vector<std::string> output_lbns_;

};

class ChainEdge final : public Edge<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;

 private:
};

class ChainGraph final : public Graph<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = default;
  ~ChainGraph() = default;

  ChainGraph(const LogicalGraph* logical_graph);

 private:
  void CollectInputAndOutputLbns();

};

std::vector<std::string> FindLbnsBetween(const ChainNode*, const ChainNode*);

} // namespace oneflow

#endif // ONEFLOW_GRAPH_CHAIN_GRAPH_H_
