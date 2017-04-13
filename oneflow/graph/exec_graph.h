#ifndef ONEFLOW_GRAPH_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_EXEC_GRAPH_H_

#include "operator/operator.h"
#include "graph/graph.h"
#include "graph/register_desc.h"

namespace oneflow {

class ExecNode;

class ExecEdge final : public Edge<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecEdge);
  ExecEdge() = default;
  ~ExecEdge() = default;

  const std::string& lbn() const { return lbn_; }
  std::string& mut_lbn() { return lbn_; }

  std::string pbn() const { return lbn2pbn(lbn_); }

 private:
  std::string lbn2pbn(const std::string& lbn) const {
    return "edge_" + std::to_string(edge_id()) + "/" + lbn;
  }

  std::string lbn_;

};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() = default;
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<const Operator>& mut_op() { return op_; }

  std::string lbn2pbn(const std::string& lbn) const {
    return "node_" + std::to_string(node_id()) + "/" + lbn;
  }

  void AddConsumedLbnRegstPair(const std::string& lbn, RegstDesc* regst);
  void AddProducedLbnRegstPair(const std::string& lbn, RegstDesc* regst);

  const std::vector<std::pair<std::string, RegstDesc*>>&
  consumed_lbn_regst_pairs() const {
    return consumed_lbn_regst_pairs_;
  }
  const std::vector<std::pair<std::string, RegstDesc*>>&
  produced_lbn_regst_pairs() const {
    return produced_lbn_regst_pairs_;
  }

 private:
  std::shared_ptr<const Operator> op_;
  std::vector<std::pair<std::string, RegstDesc*>> consumed_lbn_regst_pairs_;
  std::vector<std::pair<std::string, RegstDesc*>> produced_lbn_regst_pairs_;

};

class ExecGraph : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  virtual ~ExecGraph() = default;

  ExecNode* SoleNode() const {
    TODO();
  }

  ExecEdge* NewExecEdge(const std::string& lbn) {
    TODO();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_EXEC_GRAPH_H_
