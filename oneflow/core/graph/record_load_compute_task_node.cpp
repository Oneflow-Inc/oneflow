#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecordLoadCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  DecodeCompTaskNode* decode_node =
      static_cast<DecodeCompTaskNode*>(SoleOutEdge()->dst_node());
  task_proto->set_data_path(GetStringFromPbMessage(
      decode_node->chain_node()->SoleOp()->op_conf().decode_conf(),
      "data_dir"));
}

}  // namespace oneflow
