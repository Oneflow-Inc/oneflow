#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

void RecordLoadCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> record_regst = ProduceRegst("record", 2, 2);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("record", record_regst); }
}

void RecordLoadCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  DecodeCompTaskNode* decode_node =
      static_cast<DecodeCompTaskNode*>((*out_edges().begin())->dst_node());
  std::shared_ptr<const Operator> decode_op = decode_node->logical_node()->SoleOp();
  if (decode_op->op_conf().has_decode_ofrecord_conf()) {
    task_proto->set_record_type(RecordTypeProto::kOFRecord);
  } else {
    UNIMPLEMENTED();
  }
  std::string data_dir = decode_op->GetValFromCustomizedConf<std::string>("data_dir");
  std::string part_name_prefix =
      decode_op->GetValFromCustomizedConf<std::string>("part_name_prefix");
  int32_t part_name_suffix_length =
      decode_op->GetValFromCustomizedConf<int32_t>("part_name_suffix_length");
  std::string num = std::to_string(parallel_id());
  int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  task_proto->set_data_path(
      JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
}

}  // namespace oneflow
