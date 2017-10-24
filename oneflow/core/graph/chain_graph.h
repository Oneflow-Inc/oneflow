#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"

namespace oneflow {

class ChainEdge;

class ChainNode final : public Node<ChainNode, ChainEdge> {
 public:
  enum class Type { kForward, kBackward, kLoss };

  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  ChainNode() = default;
  ~ChainNode() = default;

  // type_
  Type type() const { return type_; }
  void set_type(Type val) { type_ = val; }

  // op_vec_
  std::shared_ptr<const Operator> SoleOp() const;
  const std::vector<std::shared_ptr<const Operator>>& op_vec() const;
  std::vector<std::shared_ptr<const Operator>>& mut_op_vec() { return op_vec_; }

  // parallel_desc_
  std::shared_ptr<const ParallelDesc> parallel_desc() const;
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc();

  // others
  std::string VisualStr() const;
  bool HasOpWithModelOrModelTmpBlob() const;

 private:
  Type type_;
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ChainNode::Type);

class ChainEdge final : public Edge<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;

  std::string VisualStr() const override;

 private:
};

class ChainGraph final : public Graph<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = default;
  ~ChainGraph() = default;

  ChainGraph(const LogicalGraph& logical_gph, bool is_train);

  const char* TypeName() const override { return "ChainGraph"; }

 private:
  void BuildFwStruct(const LogicalGraph& logical_gph);
  void BuildBpStruct();
  void BuildModelStruct(bool is_train);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
