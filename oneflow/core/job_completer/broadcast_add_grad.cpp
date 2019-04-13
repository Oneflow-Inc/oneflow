#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_add_conf());
  if (DiffLbi4BnInOp("a") != nullptr) {
    if (LogicalBlobDesc4BnInOp("out").shape() == LogicalBlobDesc4BnInOp("a").shape()) {
      *DiffLbi4BnInOp("a") = *DiffLbi4BnInOp("out");
    } else {
      OperatorConf reduce_sum_like_a_op;
      reduce_sum_like_a_op.set_name(op.op_name() + "_grad_a");
      ReduceSumLikeOpConf* reduce_sum_like_a_op_conf =
          reduce_sum_like_a_op.mutable_reduce_sum_like_conf();
      reduce_sum_like_a_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reduce_sum_like_a_op_conf->set_y("y");
      reduce_sum_like_a_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      const std::vector<int64_t> broadcast_axis_vec =
          LogicalBlobDesc4BnInOp("a").shape().BroadcastAxis(
              LogicalBlobDesc4BnInOp("out").shape().dim_vec());
      *reduce_sum_like_a_op_conf->mutable_axis() = {broadcast_axis_vec.begin(),
                                                    broadcast_axis_vec.end()};
      op_confs->push_back(reduce_sum_like_a_op);
      DiffLbi4BnInOp("a")->set_op_name(reduce_sum_like_a_op.name());
      DiffLbi4BnInOp("a")->set_blob_name("y");
    }
  }
  if (DiffLbi4BnInOp("b") != nullptr) {
    if (LogicalBlobDesc4BnInOp("out").shape() == LogicalBlobDesc4BnInOp("b").shape()) {
      *DiffLbi4BnInOp("b") = *DiffLbi4BnInOp("out");
    } else {
      OperatorConf reduce_sum_like_b_op;
      reduce_sum_like_b_op.set_name(op.op_name() + "_grad_b");
      ReduceSumLikeOpConf* reduce_sum_like_b_op_conf =
          reduce_sum_like_b_op.mutable_reduce_sum_like_conf();
      reduce_sum_like_b_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reduce_sum_like_b_op_conf->set_y("y");
      reduce_sum_like_b_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("b")));
      const std::vector<int64_t> broadcast_axis_vec =
          LogicalBlobDesc4BnInOp("b").shape().BroadcastAxis(
              LogicalBlobDesc4BnInOp("out").shape().dim_vec());
      *reduce_sum_like_b_op_conf->mutable_axis() = {broadcast_axis_vec.begin(),
                                                    broadcast_axis_vec.end()};
      op_confs->push_back(reduce_sum_like_b_op);
      DiffLbi4BnInOp("b")->set_op_name(reduce_sum_like_b_op.name());
      DiffLbi4BnInOp("b")->set_blob_name("y");
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBroadcastAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
