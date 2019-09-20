#include "oneflow/core/graph/unpack_forward_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/unpack_op.h"

namespace oneflow {

void UnpackForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void UnpackForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void UnpackForwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

void UnpackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::vector<int64_t> time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  const UnpackOp* op = dynamic_cast<UnpackOp*>(logical_node()->SoleOp().get());
  CHECK_NOTNULL(op);
  int64_t in_piece_size = in_regst->GetBlobDesc(op->BnInOp2Lbi("in"))->shape().At(0);
  int64_t unpack_num = op->GetUnpackNum();
  CHECK_EQ(0, in_piece_size % unpack_num);
  time_shape_dim_vec.push_back(unpack_num);
  *out_regst->mut_data_regst_time_shape() = std::make_shared<Shape>(std::move(time_shape_dim_vec));
}

}  // namespace oneflow
