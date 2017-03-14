#include "graph/copy_hd_transformer_graph.h"

namespace oneflow {

void CopyHDTransfmNode::FwBuildGraph() {
  auto copy_hd_task_node = of_dynamic_cast<CopyHDTaskNode*>(task_node());
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(copy_hd_task_node->CopyOpType());
  pb_op_conf.mutable_copy_op_conf()->clear_logical_blob_names();
  for (const auto& lbn : copy_hd_task_node->RelatedLbns()) {
    pb_op_conf.mutable_copy_op_conf()->add_logical_blob_names(lbn);
  }
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Transformer Node
  auto copy_node = of_dynamic_cast<CopyHDTransfmNode*>(NewTransfmNode());
  copy_node->mutable_op() = copy_op;
}

} // namespace oneflow
