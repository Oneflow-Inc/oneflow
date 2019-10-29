#include "oneflow/core/graph/every_nth_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/every_nth_op.h"

namespace oneflow {

void EveryNthCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void EveryNthCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void EveryNthCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void EveryNthCompTaskNode::InferProducedDataRegstTimeShape() {
  const EveryNthOp* op = dynamic_cast<EveryNthOp*>(this->logical_node()->SoleOp().get());
  CHECK_NOTNULL(op);
  const int64_t n = op->op_conf().every_nth_conf().n();
  const Shape* in_shape = GetSoleConsumedRegst("in")->data_regst_time_shape().get();
  DimVector dim_vec;
  CHECK_GE(in_shape->NumAxes(), 1);
  CHECK_GE(n, 1);
  if (in_shape->dim_vec().back() % n == 0) {
    dim_vec.insert(dim_vec.end(), in_shape->dim_vec().cbegin(), in_shape->dim_vec().cend() - 1);
    if (in_shape->dim_vec().back() != n) { dim_vec.push_back(in_shape->dim_vec().back() / n); }
  } else {
    dim_vec.push_back(in_shape->elem_cnt() / n);
  }
  GetProducedRegst("out")->mut_data_regst_time_shape()->reset(new Shape(dim_vec));
}

}  // namespace oneflow
