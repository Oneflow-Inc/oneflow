#include "oneflow/core/graph/pack_forward_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/pack_op.h"

namespace oneflow {

void PackForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void PackForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void PackForwardCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  exec_node->BindBnWithRegst(op->SoleIbn(), in_regst);

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);

  exec_node->InferBlobDescs(parallel_ctx());
}

void PackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  DimVector time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  const PackOp* pack_op = dynamic_cast<const PackOp*>(logical_node()->SoleOp().get());
  CHECK_NOTNULL(pack_op);
  int64_t pack_num = pack_op->op_conf().pack_conf().pack_num();
  CHECK_GT(time_shape_dim_vec.size(), 0);
  CHECK_EQ(pack_num, time_shape_dim_vec.back());
  time_shape_dim_vec.pop_back();
  *(out_regst->mut_data_regst_time_shape()) = std::make_shared<Shape>(time_shape_dim_vec);
}

}  // namespace oneflow
