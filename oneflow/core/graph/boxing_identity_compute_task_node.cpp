#include "oneflow/core/graph/boxing_identity_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void BoxingIdentityCompTaskNode::Init(const CompTaskNode* src_node, const LogicalBlobId& lbi) {
  lbi_ = lbi;
  set_logical_node(src_node->logical_node());
  *mut_parallel_ctx() = *src_node->parallel_ctx();
  set_machine_id(src_node->machine_id());
  set_thrd_id(src_node->thrd_id());
  set_area_id(src_node->area_id());
}

void BoxingIdentityCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
}

void BoxingIdentityCompTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void BoxingIdentityCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Boxing-Identity-" + NewUniqueId());
  op_conf.set_device_type(this->device_type());
  *op_conf.mutable_boxing_identity_conf()->mutable_lbi() = lbi_;
  std::shared_ptr<Operator> sole_op = ConstructOp(op_conf, &GlobalJobDesc());
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void BoxingIdentityCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
