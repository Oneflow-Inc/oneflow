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

  // Getters
  const std::string& lbn() const { return lbn_; }
  const std::string& pbn() const { return pbn_; }
  const std::string& ibn() const { return ibn_; }
  const std::string& obn() const { return obn_; }

  // Setters
  void set_lbn(const std::string& lbn);
  std::string& mut_ibn() { return ibn_; }
  std::string& mut_obn() { return obn_; }

 private:
  // various names for one blob
  std::string lbn_;
  std::string pbn_;
  std::string ibn_; // in dst_node::op
  std::string obn_; // in src_node::op

};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() = default;
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<const Operator>& mut_op() { return op_; }

  // Add pair
  void AddDtbnRegstPair(const std::string& dtbn, RegstDesc* regst);
  void AddIbnRegstPair(const std::string& ibn, RegstDesc* regst);
  void AddObnRegstPair(const std::string& obn, RegstDesc* regst);
  void AddMbnRegstPair(const std::string& mbn, RegstDesc* regst);
  void AddMtbnRegstPair(const std::string& mtbn, RegstDesc* regst);

  // Get Pairs
  #define DEFINE_PAIRS_GETTER(getter) \
  const std::vector<std::pair<std::string, RegstDesc*>>& getter() const { \
    return getter##_; \
  }

  DEFINE_PAIRS_GETTER(dtbn_regst_pairs);
  DEFINE_PAIRS_GETTER(ibn_regst_pairs);
  DEFINE_PAIRS_GETTER(obn_regst_pairs);
  DEFINE_PAIRS_GETTER(mbn_regst_pairs);
  DEFINE_PAIRS_GETTER(mtbn_regst_pairs);

  #undef DEFINE_PAIRS_GETTER

 private:
  std::shared_ptr<const Operator> op_;
  std::vector<std::pair<std::string, RegstDesc*>> dtbn_regst_pairs_;
  std::vector<std::pair<std::string, RegstDesc*>> ibn_regst_pairs_;
  std::vector<std::pair<std::string, RegstDesc*>> obn_regst_pairs_;
  std::vector<std::pair<std::string, RegstDesc*>> mbn_regst_pairs_;
  std::vector<std::pair<std::string, RegstDesc*>> mtbn_regst_pairs_;

};

class ExecGraph final : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  ~ExecGraph() = default;

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
