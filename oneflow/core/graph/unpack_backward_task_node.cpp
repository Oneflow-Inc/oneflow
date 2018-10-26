#include "oneflow/core/graph/unpack_backward_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void UnpackBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("in_diff", false);
  BindEdgeWithProducedRegst(SoleOutEdge(), "in_diff");
}

void UnpackBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    if (edge->src_node()->GetTaskType() == TaskType::kUnpackForward) {
      ConsumeRegst("in", edge->src_node()->GetSoleConsumedRegst("in"));
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }
}

void UnpackBackwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleOdbn(), GetSoleConsumedRegst("out_diff"));

  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  in_diff_regst->AddLbi(op->BnInOp2Lbi(op->SoleIdbn()));
  exec_node->BindBnWithRegst(op->SoleIdbn(), in_diff_regst);
  in_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("in").get());
}

void UnpackBackwardCompTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("in_diff")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
}

}  // namespace oneflow
