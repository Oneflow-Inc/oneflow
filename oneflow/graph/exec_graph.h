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
    return "edge_id_" + std::to_string(edge_id()) + "/" + lbn;
  }

  std::string lbn_;

};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() = default;
  ~ExecNode() = default;

  std::shared_ptr<Operator> op() const { return op_; }
  std::shared_ptr<Operator>& mut_op() { return op_; }

  std::string lbn2pbn(const std::string& lbn) const {
    return "node_id_" + std::to_string(node_id()) + "/" + lbn;
  }

  void AddConsumedLbnRegiPair(const std::string& lbn, RegisterDesc* register_desc);
  void AddProducedLbnRegiPair(const std::string& lbn, RegisterDesc* register_desc);

  const std::vector<std::pair<std::string, RegisterDesc*>>& produced_lbn_regi_pairs() const {
    return produced_lbn_regi_pairs_;
  }

 private:
  std::shared_ptr<Operator> op_;
  std::vector<std::pair<std::string, RegisterDesc*>> consumed_lbn_regi_pairs_;
  std::vector<std::pair<std::string, RegisterDesc*>> produced_lbn_regi_pairs_;

};

class ExecGraph : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  virtual ~ExecGraph() = default;

  ExecNode* NewExecNode() {
    LOG(FATAL) << "TODO";
  }
  ExecEdge* NewExecEdge(const std::string& lbn) {
    LOG(FATAL) << "TODO";
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_EXEC_GRAPH_H_
