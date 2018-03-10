#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecordLoadCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  DecodeCompTaskNode* decode_node =
      static_cast<DecodeCompTaskNode*>(SoleOutEdge()->dst_node());
  std::string data_dir =
      decode_node->chain_node()->SoleOp()->GetStringFromCustomizedConf(
          "data_dir");
  std::string part_name_prefix =
      decode_node->chain_node()->SoleOp()->GetStringFromCustomizedConf(
          "part_name_prefix");
  int32_t part_name_suffix_length =
      decode_node->chain_node()->SoleOp()->GetInt32FromCustomizedConf(
          "part_name_suffix_length");
  std::string num = std::to_string(parallel_id());
  std::string data_path = data_dir + "/" + part_name_prefix;
  FOR_RANGE(size_t, i, 0, part_name_suffix_length - num.length()) {
    data_path += "0";
  }
  data_path += num;
  task_proto->set_data_path(data_path);
}

}  // namespace oneflow
