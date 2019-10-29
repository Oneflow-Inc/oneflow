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

  const auto& related_unpack_consumed_regsts = related_unpack_->consumed_regsts();
  CHECK_EQ(1, related_unpack_consumed_regsts.size());
  CHECK_EQ(1, (*related_unpack_consumed_regsts.begin()).second.size());
  std::shared_ptr<RegstDesc> related_unpack_consumed_regst =
      (*related_unpack_consumed_regsts.begin()).second.front();

  auto GetBlobDescFromRegst = [&](std::shared_ptr<RegstDesc> regst) -> const BlobDesc* {
    const BlobDesc* ret_blob = nullptr;
    regst->ForEachLbi([&](const LogicalBlobId& lbi) {
      if (ret_blob == nullptr) { ret_blob = regst->GetBlobDesc(lbi); }
    });
    return ret_blob;
  };
  const BlobDesc* related_unpack_in_blob = GetBlobDescFromRegst(related_unpack_consumed_regst);
  const BlobDesc* in_blob = GetBlobDescFromRegst(in_regst);
  BlobDesc* out_blob = out_regst->MutSoleBlobDesc();

  *out_blob = *in_blob;
  const PackOp* pack_op = dynamic_cast<const PackOp*>(op.get());
  CHECK_NOTNULL(pack_op);
  CHECK_EQ(pack_op->GetPackNum(), related_unpack_in_blob->shape().At(0) / in_blob->shape().At(0));
  out_blob->mut_shape().Set(0, related_unpack_in_blob->shape().At(0));
}

void PackForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  DimVector time_shape_dim_vec(in_regst->data_regst_time_shape()->dim_vec());

  const PackOp* pack_op = dynamic_cast<const PackOp*>(logical_node()->SoleOp().get());
  CHECK_NOTNULL(pack_op);
  int64_t pack_num = pack_op->GetPackNum();
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
