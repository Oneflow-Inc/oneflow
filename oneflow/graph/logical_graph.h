#ifndef ONEFLOW_GRAPH_LOGICAL_GRAPH_H_
#define ONEFLOW_GRAPH_LOGICAL_GRAPH_H_

#include <memory>
#include "graph/graph.h"
#include "operator/operator.h"
#include "job/dlnet_conf.pb.h"
#include "job/strategy.pb.h"
#include "job/parallel_desc.h"

namespace oneflow {

class LogicalEdge;

class LogicalNode final : public Node<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalNode);
  LogicalNode() = default;
  ~LogicalNode() = default;

  std::shared_ptr<const Operator> op() const {
    return op_;
  }
  std::shared_ptr<const Operator>& mut_op() {
    return op_;
  }

  std::shared_ptr<const ParallelDesc> parallel_desc() const {
    return parallel_desc_;
  }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() {
    return parallel_desc_;
  }

  bool IsLossNode() const { return op_->IsLossOp(); }

 private:
  std::shared_ptr<const Operator> op_;
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
               const Strategy& strategy_conf);

 private:
  void NaiveBuildGraphStruct(
      const DLNetConf& dl_net_conf,
      HashMap<LogicalEdge*, std::string>* edge2lbn);
  void FillNodeWithParallelDesc(const Strategy& strategy_conf);

  struct CloneInfo {
    std::shared_ptr<const Operator> clone_op;
    LogicalNode* pred_node;
    std::vector<LogicalEdge*> edges;
  };
  void AddCloneNodes(
      const HashMap<LogicalEdge*, std::string>& edge2lbn);
  void CollectCloneInfos(
      std::vector<CloneInfo>* clone_infos,
      const HashMap<LogicalEdge*, std::string>& edge2lbn);
  void AddOneCloneNode(const CloneInfo& clone_info);


};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_LOGICAL_GRAPH_H_
