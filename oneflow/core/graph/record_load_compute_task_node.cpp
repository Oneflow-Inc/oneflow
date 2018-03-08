#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"

namespace oneflow {

void RecordLoadCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  DecodeCompTaskNode* task_node =
      static_cast<DecodeCompTaskNode*>(SoleOutEdge()->dst_node());
  task_proto->set_data_dir(task_node->GetDataDir());
}

}  // namespace oneflow
