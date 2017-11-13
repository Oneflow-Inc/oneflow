#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LogicalEdge;

class LogicalNode final : public Node<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalNode);
  LogicalNode() = default;
  ~LogicalNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<Operator>& mut_op() { return op_; }

  std::shared_ptr<const ParallelDesc> parallel_desc() const {
    return parallel_desc_;
  }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() {
    return parallel_desc_;
  }

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
  ~LogicalGraph() = default;

  OF_SINGLETON(LogicalGraph);

  const char* TypeName() const override { return "LogicalGraph"; }
  const LogicalNode* GetProducerNode(const std::string& lbn);

 private:
  LogicalGraph();
  void NaiveBuildGraphStruct(HashMap<LogicalEdge*, std::string>* edge2lbn,
                             HashMap<LogicalEdge*, std::string>* edge2ibn);
  void FillNodeWithParallelDesc();

  struct CloneInfo {
    std::shared_ptr<Operator> clone_op;
    LogicalNode* pred_node;
    std::vector<LogicalEdge*> edges;
  };
  void AddCloneNodes(const HashMap<LogicalEdge*, std::string>& edge2lbn,
                     const HashMap<LogicalEdge*, std::string>& edge2ibn);
  void CollectCloneInfos(std::vector<CloneInfo>* clone_infos,
                         const HashMap<LogicalEdge*, std::string>& edge2lbn);
  void AddOneCloneNode(const CloneInfo& clone_info,
                       const HashMap<LogicalEdge*, std::string>& edge2ibn);

  HashMap<std::string, const LogicalNode*> lbn2producer_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
