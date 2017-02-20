#include "dag/logical_dag.h"

namespace oneflow {

void LogicalDag::Init(const std::string& dag_name,
                      const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Dag::Init(dag_name);
  BuildDagStruct(dl_net_conf);
  FillNodeWithPlacement(strategy_conf);
}

void BuildDagStruct(const DLNetConf& dl_net_conf) {
  for (int layer_i = 0; layer_i < dl_net_conf.layer_conf_size(); ++layer_i) {
    const LayerConf& cur_layer_conf = dl_net_conf.layer_conf(layer_i);
    // TODO
  }
}

void FillNodeWithPlacement(const Strategy& strategy_conf) {
  // TODO
}

} // namespace oneflow
