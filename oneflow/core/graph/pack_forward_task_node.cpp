#include "oneflow/core/graph/pack_forward_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void PackForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() != "PackBackward") { BindEdgeWithProducedRegst(edge, "out"); }
  }
}

void PackForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
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

  const auto& related_unpack_consumed_regsts = related_unpack_->consumed_regsts();
  CHECK_EQ(1, related_unpack_consumed_regsts.size());
  CHECK_EQ(1, (*related_unpack_consumed_regsts.begin()).second.size());
  std::shared_ptr<RegstDesc> related_unpack_consumed_regst =
      (*related_unpack_consumed_regsts.begin()).second.front();
  const BlobDesc* related_unpack_in_blob = related_unpack_consumed_regst->SoleBlobDesc();
  const BlobDesc* in_blob = in_regst->SoleBlobDesc();
  BlobDesc* out_blob = out_regst->MutSoleBlobDesc();
  *out_blob = *in_blob;
  CHECK_EQ(op->op_conf().pack_conf().pack_num(),
           related_unpack_in_blob->shape().At(0) / in_blob->shape().At(0));
  out_blob->mut_shape().Set(0, related_unpack_in_blob->shape().At(0));
  if (out_blob->has_dim0_valid_num_field()) {
    out_blob->mut_dim0_inner_shape() = related_unpack_in_blob->dim0_inner_shape();
  }
}

void PackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::vector<int64_t> time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  int64_t pack_num = op->op_conf().pack_conf().pack_num();
  CHECK_GT(time_shape_dim_vec.size(), 0);
  CHECK_EQ(pack_num, time_shape_dim_vec.back());
  time_shape_dim_vec.pop_back();

  std::shared_ptr<RegstDesc> related_unpack_consumed_regst =
      (*related_unpack_->consumed_regsts().begin()).second.front();
  std::shared_ptr<Shape> pack_out_time_shape =
      std::make_shared<Shape>(std::move(time_shape_dim_vec));
  CHECK_EQ(*pack_out_time_shape, *(related_unpack_consumed_regst->data_regst_time_shape()));
  *(out_regst->mut_data_regst_time_shape()) = pack_out_time_shape;
}

}  // namespace oneflow
