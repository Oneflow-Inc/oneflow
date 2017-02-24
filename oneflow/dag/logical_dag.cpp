#include "dag/logical_dag.h"
#include "layer/layer_desc_factory.h"

namespace oneflow {

void LogicalDag::Init(const std::string& dag_name,
                      const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Dag::Init(dag_name);
  BuildDagStruct(dl_net_conf);
  FillNodeWithPlacement(strategy_conf);
}

void LogicalDag::BuildDagStruct(const DLNetConf& dl_net_conf) {
  for (int layer_i = 0; layer_i < dl_net_conf.layer_conf_size(); ++layer_i) {
    const LayerConf& cur_layer_conf = dl_net_conf.layer_conf(layer_i);
    // Construct op node
    LogicalOpNode* cur_op_node = NewLogicalOpNode();
    cur_op_node->mutable_layer_desc() =
        LayerDescFactory::singleton().ConstructLayerDesc(cur_layer_conf);
    // Construct input data node
    for (const std::string& input_blob_name
        : cur_op_node->layer_desc().data_blob_desc_set().input_blob_names()) {
      input_blob_name.find('/');
    }
    // Construct output data node
  }
}

void FillNodeWithPlacement(const Strategy& strategy_conf) {
  // TODO
}

} // namespace oneflow
