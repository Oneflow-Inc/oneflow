#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_

#include <memory>
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/conf/dlnet_conf.pb.h"
#include "oneflow/core/conf/strategy.pb.h"
#include "oneflow/core/compile/parallel_desc.h"

namespace oneflow {

class LogicalEdge;

class LogicalNode final : public Node<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalNode);
  LogicalNode() = default;
  ~LogicalNode() = default;

  std::shared_ptr<Operator> op() const {
    return op_;
  }
  std::shared_ptr<Operator>& mut_op() {
    return op_;
  }

  std::shared_ptr<const ParallelDesc> parallel_desc() const {
    return parallel_desc_;
  }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() {
    return parallel_desc_;
  }

  bool IsLossNode() const { return op_->IsLossOp(); }

  std::string VisualStr() const override { return op_->op_name(); }

 private:
  std::shared_ptr<Operator> op_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;

};

class LogicalEdge final : public Edge<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalEdge);
  LogicalEdge() = default;
  ~LogicalEdge() = default;

 private:
};

class LogicalGraph final : public Graph<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalGraph);
  LogicalGraph() = delete;
  ~LogicalGraph() = default;

  LogicalGraph(const DLNetConf& dl_net_conf,
               const Strategy& strategy_conf,
               const std::string& dot_filepath);

 private:
  void NaiveBuildGraphStruct(
      const DLNetConf& dl_net_conf,
      HashMap<LogicalEdge*, std::string>* edge2lbn,
      HashMap<LogicalEdge*, std::string>* edge2ibn);
  void FillNodeWithParallelDesc(const Strategy& strategy_conf);

  struct CloneInfo {
    std::shared_ptr<Operator> clone_op;
    LogicalNode* pred_node;
    std::vector<LogicalEdge*> edges;
  };
  void AddCloneNodes(
      const HashMap<LogicalEdge*, std::string>& edge2lbn,
      const HashMap<LogicalEdge*, std::string>& edge2ibn);
  void CollectCloneInfos(
      std::vector<CloneInfo>* clone_infos,
      const HashMap<LogicalEdge*, std::string>& edge2lbn);
  void AddOneCloneNode(
      const CloneInfo& clone_info,
      const HashMap<LogicalEdge*, std::string>& edge2ibn);


};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
