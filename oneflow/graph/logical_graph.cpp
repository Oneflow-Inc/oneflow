#include "graph/logical_graph.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void LogicalGraph::Init(const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Graph::Init();
  BuildGraphStruct(dl_net_conf);
  FillNodeWithParallelDesc(strategy_conf);
}

// "BlobNameInGraph" = "OpName/BlobNameInOp"
// "BlobNameInGraphIf" means Blob is the input of op
// "BlobNameInGraphOf" means Blob is the output of op
// "LogicalBlobName" = "BlobNameInGraphOf"

static std::string BlobNameInGraph2BlobNameInOp(
    const std::string& blob_name_in_graph) {
  size_t slash_pos = blob_name_in_graph.find('/');
  CHECK(slash_pos != std::string::npos);
  return blob_name_in_graph.substr(slash_pos + 1);
}

static std::string BlobNameInOp2BlobNameInGraph(
    const std::string& op_name,
    const std::string blob_name_in_op) {
  return op_name + "/" + blob_name_in_op;
}

void LogicalGraph::BuildGraphStruct(const DLNetConf& dl_net_conf) {
  std::unordered_map<std::string, LogicalNode*> logical_blob_name2node;
  // Process Op
  for (int op_i = 0; op_i < dl_net_conf.op_conf_size(); ++op_i) {
    const OperatorConf& cur_op_conf = dl_net_conf.op_conf(op_i);
    // Construct cur node
    LogicalNode* cur_node = NewLogicalNode();
    cur_node->mutable_op_ptr() =
        OperatorFactory::singleton().ConstructOp(cur_op_conf);
    // Connect input node
    for (const std::string& blob_name_in_op
        : cur_node->op().data_blob_desc_set().input_blob_names()) {
      std::string logical_blob_name =
          GetStringValueFromPbMessage(cur_op_conf, blob_name_in_op);
      LogicalNode* pred_node = logical_blob_name2node.at(logical_blob_name);
      Connect(pred_node, NewLogicalEdge(), cur_node);
    }
    // Construct output
    for (const std::string& blob_name_in_op
        : cur_node->op().data_blob_desc_set().output_blob_names()) {
      std::string logical_blob_name =
          BlobNameInOp2BlobNameInGraph(cur_node->op().op_name(),
                                       blob_name_in_op);
      logical_blob_name2node.emplace(logical_blob_name, cur_node);
    }
  }
  logical_blob_name2node.clear();
  // Post Processing
  UpdateStartAndStop();
}

void LogicalGraph::FillNodeWithParallelDesc(const Strategy& strategy_conf) {
  std::unordered_map<std::string, LogicalNode*> op_name2node;
  for (const std::unique_ptr<Node>& node : node_vec()) {
    auto logical_node_ptr = of_dynamic_cast<LogicalNode*> (node.get());
    std::string op_name = logical_node_ptr->op().op_name();
    bool emplace_success =
        op_name2node.emplace(op_name, logical_node_ptr).second;
    CHECK_EQ(emplace_success, true);
  }
  for (int gid = 0; gid < strategy_conf.placement_group_vec_size(); ++gid) {
    const PlacementGroup& cur_group = strategy_conf.placement_group_vec(gid);
    for (int li = 0; li < cur_group.op_name_vec_size(); ++li) {
      const std::string& op_name = cur_group.op_name_vec(li);
      auto it = op_name2node.find(op_name);
      CHECK(it != op_name2node.end());
      ParallelDesc* parallel_desc_raw_ptr = new ParallelDesc;
      parallel_desc_raw_ptr->Init(cur_group.parallel_conf());
      it->second->mutable_parallel_desc_ptr().reset(parallel_desc_raw_ptr);
    }
  }
}

} // namespace oneflow
