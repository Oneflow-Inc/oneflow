#include "dag/logical_dag.h"
#include "glog/logging.h"
#include "layer/layer_desc_factory.h"

namespace oneflow {

void LogicalDag::Init(const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Dag::Init();
  BuildDagStruct(dl_net_conf);
  FillNodeWithParallelDesc(strategy_conf);
}

// "BlobNameInDag" = "LayerName/BlobNameInLayer"
// BlobNameInDagIf means Blob is the input of layer
// BlobNameInDagOf means Blob is the output of layer

static std::string BlobNameInDag2BlobNameInLayer(
    const std::string& blob_name_in_dag) {
  size_t slash_pos = blob_name_in_dag.find('/');
  CHECK(slash_pos != std::string::npos);
  return blob_name_in_dag.substr(slash_pos + 1);
}

void LogicalDag::BuildDagStruct(const DLNetConf& dl_net_conf) {
  std::unordered_map<std::string, LogicalNode*> blob_name_indag_of2node;
  // Process Layer
  for (int layer_i = 0; layer_i < dl_net_conf.layer_conf_size(); ++layer_i) {
    const LayerConf& cur_layer_conf = dl_net_conf.layer_conf(layer_i);
    // Construct cur node
    LogicalNode* cur_node = NewLogicalNode();
    cur_node->mutable_layer_desc_ptr() =
        LayerDescFactory::singleton().ConstructLayerDesc(cur_layer_conf);
    // Connect input node
    for (const std::string& blob_name_in_dag_if
        : cur_node->layer_desc().data_blob_desc_set().input_blob_names()) {
      std::string blob_name_in_layer =
          BlobNameInDag2BlobNameInLayer(blob_name_in_dag_if);
      std::string blob_name_indag_of =
          GetStringValueFromPbMessage(cur_layer_conf, blob_name_in_layer);
      auto pre_node_it = blob_name_indag_of2node.find(blob_name_indag_of);
      CHECK(pre_node_it != blob_name_indag_of2node.end());
      ConnectTwoNode(pre_node_it->second, cur_node);
    }
    // Construct output
    for (const std::string& blob_name_indag_of
        : cur_node->layer_desc().data_blob_desc_set().output_blob_names()) {
      bool insert_success =
          blob_name_indag_of2node.emplace(blob_name_indag_of, cur_node).second;
      CHECK_EQ(insert_success, true);
    }
  }
  blob_name_indag_of2node.clear();
  // Post Processing
  ConnectStartAndStop();
}

void LogicalDag::FillNodeWithParallelDesc(const Strategy& strategy_conf) {
  std::unordered_map<std::string, LogicalNode*> layer_name2node;
  for (const std::unique_ptr<DagNode>& node : node_vec()) {
    auto logical_node_ptr = of_dynamic_cast<LogicalNode*> (node.get());
    std::string layer_name = logical_node_ptr->layer_desc().layer_name();
    bool emplace_success =
        layer_name2node.emplace(layer_name, logical_node_ptr).second;
    CHECK_EQ(emplace_success, true);
  }
  for (int gid = 0; gid < strategy_conf.placement_group_vec_size(); ++gid) {
    const PlacementGroup& cur_group = strategy_conf.placement_group_vec(gid);
    for (int li = 0; li < cur_group.layer_name_vec_size(); ++li) {
      const std::string& layer_name = cur_group.layer_name_vec(li);
      auto it = layer_name2node.find(layer_name);
      CHECK(it != layer_name2node.end());
      ParallelDesc* parallel_desc_raw_ptr = new ParallelDesc;
      parallel_desc_raw_ptr->Init(cur_group.parallel_conf());
      it->second->mutable_parallel_desc_ptr().reset(parallel_desc_raw_ptr);
    }
  }
}

} // namespace oneflow
