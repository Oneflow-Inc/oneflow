#include "dag/logical_dag.h"
#include "glog/logging.h"
#include "layer/layer_desc_factory.h"

namespace oneflow {

void LogicalDag::Init(const std::string& dag_name,
                      const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Dag::Init(dag_name);
  BuildDagStruct(dl_net_conf);
  FillNodeWithParallelDesc(strategy_conf);
}

// BlobNameInDag = LayerName/BlobNameInLayer
// BlobNameInDagIf means Blob is the input of layer
// BlobNameInDagOf means Blob is the output of layer
static std::string BlobNameInDag2BlobNameInLayer(
    const std::string& blob_name_in_dag) {
  size_t slash_pos = blob_name_in_dag.find('/');
  CHECK(slash_pos != std::string::npos);
  return blob_name_in_dag.substr(slash_pos + 1);
}

void LogicalDag::BuildDagStruct(const DLNetConf& dl_net_conf) {
  // This function only execute few times, so it is ok to declare it
  std::unordered_map<std::string, LogicalDataNode*> blob_name_indag_of2ptr;
  // Process Layer
  for (int layer_i = 0; layer_i < dl_net_conf.layer_conf_size(); ++layer_i) {
    const LayerConf& cur_layer_conf = dl_net_conf.layer_conf(layer_i);
    // Construct op node
    LogicalOpNode* cur_op_node = NewLogicalOpNode();
    cur_op_node->mutable_layer_desc_ptr() =
        LayerDescFactory::singleton().ConstructLayerDesc(cur_layer_conf);
    // Connect input data node
    for (const std::string& blob_name_in_dag_if
        : cur_op_node->layer_desc().data_blob_desc_set().input_blob_names()) {
      std::string blob_name_in_layer =
          BlobNameInDag2BlobNameInLayer(blob_name_in_dag_if);
      std::string blob_name_indag_of =
          GetStringValueFromPbMessage(cur_layer_conf, blob_name_in_layer);
      auto data_node_it = blob_name_indag_of2ptr.find(blob_name_indag_of);
      CHECK(data_node_it != blob_name_indag_of2ptr.end());
      cur_op_node->AddPredecessor(data_node_it->second);
    }
    // Construct and connect output data node
    for (const std::string& blob_name_indag_of
        : cur_op_node->layer_desc().data_blob_desc_set().output_blob_names()) {
      LogicalDataNode* data_node = NewLogicalDataNode();
      bool insert_success =
          blob_name_indag_of2ptr.emplace(blob_name_indag_of, data_node).second;
      CHECK_EQ(insert_success, true);
      data_node->AddPredecessor(cur_op_node);
    }
  }
  blob_name_indag_of2ptr.clear();
  // Post Processing
  ConnectStartAndStop();
  //ConnectLogicalOpNodePtr();
  ConnectOpNodeExtraPtr(this);
}

void LogicalDag::FillNodeWithParallelDesc(const Strategy& strategy_conf) {
  // This function only execute few times, so it is ok to declare it
  std::unordered_map<std::string, LogicalOpNode*> layer_name2op_node;
  for (const std::unique_ptr<OpNode>& op_node : op_node_vec()) {
    auto logical_op_node_ptr = of_dynamic_cast<LogicalOpNode*> (op_node.get());
    std::string layer_name = logical_op_node_ptr->layer_desc().layer_name();
    bool emplace_success =
        layer_name2op_node.emplace(layer_name, logical_op_node_ptr).second;
    CHECK_EQ(emplace_success, true);
  }
  for (int gid = 0; gid < strategy_conf.placement_group_vec_size(); ++gid) {
    const PlacementGroup& cur_group = strategy_conf.placement_group_vec(gid);
    for (int li = 0; li < cur_group.layer_name_vec_size(); ++li) {
      const std::string& layer_name = cur_group.layer_name_vec(li);
      auto it = layer_name2op_node.find(layer_name);
      CHECK(it != layer_name2op_node.end());
      ParallelDesc* parallel_desc_raw_ptr = new ParallelDesc;
      parallel_desc_raw_ptr->Init(cur_group.parallel_conf());
      it->second->mutable_parallel_desc_ptr().reset(parallel_desc_raw_ptr);
    }
  }
}

/*
void LogicalDag::ConnectLogicalOpNodePtr() {
  for (const std::unique_ptr<OpNode>& op_node : op_node_vec()) {
    auto cur_node = of_dynamic_cast<LogicalOpNode*> (op_node.get());
    for (const DagNode* data_pre_node : cur_node->predecessors()) {
      for (const DagNode* op_pre_node : data_pre_node->predecessors()) {
        cur_node->mutable_op_predecessors().insert(
            of_dynamic_cast<const LogicalOpNode*> (op_pre_node));
      }
    }
    for (const DagNode* data_next_node : cur_node->successors()) {
      for (const DagNode* op_next_node : data_next_node->successors()) {
        cur_node->mutable_op_successors().insert(
            of_dynamic_cast<const LogicalOpNode*> (op_next_node));
      }
    }
  }
}*/

} // namespace oneflow
