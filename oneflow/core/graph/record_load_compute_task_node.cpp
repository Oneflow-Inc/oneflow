#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

std::shared_ptr<const Operator> RecordLoadCompTaskNode::GetRelatedDecodeOp() {
  DecodeCompTaskNode* decode_node =
      static_cast<DecodeCompTaskNode*>((*out_edges().begin())->dst_node());
  return decode_node->logical_node()->SoleOp();
}

void RecordLoadCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> record_regst = ProduceRegst("record", 2, 2);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("record", record_regst); }
}

void RecordLoadCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> record_regst = GetProducedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    record_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, record_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void RecordLoadCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  int32_t data_part_num = Global<JobDesc>::Get()->DataPartNum();
  std::shared_ptr<const Operator> decode_op = GetRelatedDecodeOp();
  std::string data_dir = decode_op->GetValFromCustomizedConf<std::string>("data_dir");
  std::string part_name_prefix =
      decode_op->GetValFromCustomizedConf<std::string>("part_name_prefix");
  int32_t part_name_suffix_length =
      decode_op->GetValFromCustomizedConf<int32_t>("part_name_suffix_length");
  int32_t parallel_num = parallel_ctx()->parallel_num();
  CHECK_LE(parallel_num, data_part_num);
  BalancedSplitter bs(data_part_num, parallel_num);
  Range range = bs.At(parallel_id());
  FOR_RANGE(int32_t, part_id, range.begin(), range.end()) {
    std::string num = std::to_string(part_id);
    int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
    task_proto->add_data_path(
        JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
  }
}

}  // namespace oneflow
