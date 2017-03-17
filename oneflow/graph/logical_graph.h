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
  DISALLOW_COPY_AND_MOVE(LogicalNode);
  LogicalNode() = default;
  ~LogicalNode() = default;

  void Init() {
    Node<LogicalNode, LogicalEdge>::Init();
  }

  const Operator& op() const {
    return *op_ptr_;
  }
  std::shared_ptr<const Operator> op_ptr() const {
    return op_ptr_;
  }
  std::shared_ptr<const Operator>& mutable_op_ptr() {
    return op_ptr_;
  }

  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

 private:
  std::shared_ptr<const Operator> op_ptr_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class LogicalEdge final : public Edge<LogicalNode, LogicalEdge> {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalEdge);
  LogicalEdge() = default;
  ~LogicalEdge() = default;
  
  void Init() {
    Edge<LogicalNode, LogicalEdge>::Init();
  }

 private:
};

class LogicalGraph final : public Graph<LogicalNode, LogicalEdge> {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalGraph);
  LogicalGraph() = default;
  ~LogicalGraph() = default;

  void Init(const DLNetConf& dl_net_conf,
            const Strategy& strategy_conf);

 private:
  void BuildGraphStruct(const DLNetConf& dl_net_conf);
  void FillNodeWithParallelDesc(const Strategy& strategy_conf);

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_LOGICAL_GRAPH_H_
