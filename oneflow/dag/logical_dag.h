#ifndef ONEFLOW_DAG_LOGICAL_DAG_H
#define ONEFLOW_DAG_LOGICAL_DAG_H

#include <memory>
#include "dag/dag.h"
#include "dag/logical_data_node.h"
#include "dag/logical_op_node.h"
#include "job/dlnet_conf.pb.h"
#include "job/strategy.pb.h"

namespace oneflow {

class LogicalDag : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalDag);
  LogicalDag() = default;
  ~LogicalDag() = default;

  void Init(const std::string& dag_name,
            const DLNetConf& dl_net_conf,
            const Strategy& strategy_conf);

 private:
  void BuildDagStruct(const DLNetConf& dl_net_conf);
  void FillNodeWithPlacement(const Strategy& strategy_conf);

  LogicalDataNode* NewLogicalDataNode() {
    std::unique_ptr<LogicalDataNode> new_node(new LogicalDataNode);
    new_node->Init();
    logical_data_node_vec_.push_back(std::move(new_node));
    return logical_data_node_vec_.back().get();
  }

  LogicalOpNode* NewLogicalOpNode() {
    std::unique_ptr<LogicalOpNode> new_node(new LogicalOpNode);
    new_node->Init();
    logical_op_node_vec_.push_back(std::move(new_node));
    return logical_op_node_vec_.back().get();
  }

  std::vector<std::unique_ptr<LogicalDataNode>> logical_data_node_vec_;
  std::vector<std::unique_ptr<LogicalOpNode>> logical_op_node_vec_;

};

} // namespace oneflow

#endif // ONEFLOW_DAG_LOGICAL_DAG_H
