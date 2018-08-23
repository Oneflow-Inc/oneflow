#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OpEdge;

class OpNode final : public Node<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpNode);
  OpNode() = default;
  ~OpNode() override = default;

  const std::vector<std::shared_ptr<Operator>>& op_vec() const { return op_vec_; }
  std::vector<std::shared_ptr<Operator>>& mut_op_vec() { return op_vec_; }

  std::shared_ptr<const ParallelDesc> parallel_desc() const { return parallel_desc_; }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() { return parallel_desc_; }

  std::string VisualStr() const override;

 private:
  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  OpEdge() = default;
  ~OpEdge() override = default;

  const std::vector<LogicalBlobId>& lbi_vec() const { return lbi_vec_; };
  std::vector<LogicalBlobId>& mut_lbi_vec() { return lbi_vec_; }

 private:
  std::vector<LogicalBlobId> lbi_vec_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  OpGraph();
  OpGraph(const DLNetConf& net_conf, const Placement& placement);
  ~OpGraph() override = default;

  const char* TypeName() const override { return "OpGraph"; }

 private:
  void BuildFwStruct(const DLNetConf& net_conf, const Placement& placement);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
