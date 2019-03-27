#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_reduce_sum_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf broadcast_like_op;
    broadcast_like_op.set_name(op.op_name() + "_grad");
    BroadcastLikeOpConf* broadcast_like_op_conf = broadcast_like_op.mutable_broadcast_like_conf();
    broadcast_like_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    broadcast_like_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    broadcast_like_op_conf->set_y("y");
    const ReduceSumOpConf& reduce_sum_op_conf = op.op_conf().reduce_sum_conf();
    broadcast_like_op_conf->mutable_axis()->CopyFrom(reduce_sum_op_conf.axis());
    op_confs->push_back(broadcast_like_op);
    DiffLbi4BnInOp("in")->set_op_name(broadcast_like_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("y");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReduceSumConf, &GenerateBackwardOpConf);

}  // namespace oneflow
