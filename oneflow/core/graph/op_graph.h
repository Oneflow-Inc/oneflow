#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OpEdge;

class OpNode final : public Node<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpNode);
  virtual ~OpNode() = default;

  // op_vec_
  // std::shared_ptr<Op> SoleOp() const;
  const std::vector<std::shared_ptr<Operator>>& op_vec() const { return op_vec_; }
  std::vector<std::shared_ptr<Operator>>& mut_op_vec() { return op_vec_; }

  // parallel_desc_
  std::shared_ptr<const ParallelDesc> parallel_desc() const { return parallel_desc_; }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() { return parallel_desc_; }

 private:
  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  OpEdge() = default;
  ~OpEdge() = default;

 private:
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  OpGraph();
  ~OpGraph() = default;

 private:
  void BuildFwStruct();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
