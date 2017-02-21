#ifndef ONEFLOW_DAG_LOGICAL_DAG_H
#define ONEFLOW_DAG_LOGICAL_DAG_H

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

};

} // namespace oneflow

#endif // ONEFLOW_DAG_LOGICAL_DAG_H
