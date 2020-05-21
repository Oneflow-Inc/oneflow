#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_add_conf());
  if (DiffLbi4BnInOp("a") != nullptr) {
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(LogicalBlobDesc4BnInOp("a").shape()),
                                LogicalBlobDesc4BnInOp("out").shape().NumAxes());
    if (LogicalBlobDesc4BnInOp("out").shape() == LogicalBlobDesc4BnInOp("a").shape()) {
      *DiffLbi4BnInOp("a") = *DiffLbi4BnInOp("out");
    } else if (LogicalBlobDesc4BnInOp("out").shape() == left_extended_shape) {
      OperatorConf reshape_like_a_op;
      reshape_like_a_op.set_name(op.op_name() + "_grad_a_reshape_like");
      ReshapeLikeOpConf* reshape_like_a_op_conf = reshape_like_a_op.mutable_reshape_like_conf();
      reshape_like_a_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reshape_like_a_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      reshape_like_a_op_conf->set_y("y");
      op_confs->push_back(reshape_like_a_op);
      DiffLbi4BnInOp("a")->set_op_name(reshape_like_a_op.name());
      DiffLbi4BnInOp("a")->set_blob_name("y");
    } else {
      OperatorConf reduce_sum_like_a_op;
      reduce_sum_like_a_op.set_name(op.op_name() + "_grad_a");
      ReduceSumLikeOpConf* reduce_sum_like_a_op_conf =
          reduce_sum_like_a_op.mutable_reduce_sum_like_conf();
      reduce_sum_like_a_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reduce_sum_like_a_op_conf->set_y("y");
      reduce_sum_like_a_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      const AxisVector& broadcast_axis_vec =
          left_extended_shape.Axes4BroadcastTo(LogicalBlobDesc4BnInOp("out").shape());
      *reduce_sum_like_a_op_conf->mutable_axis() = {broadcast_axis_vec.begin(),
                                                    broadcast_axis_vec.end()};
      op_confs->push_back(reduce_sum_like_a_op);
      DiffLbi4BnInOp("a")->set_op_name(reduce_sum_like_a_op.name());
      DiffLbi4BnInOp("a")->set_blob_name("y");
    }
  }
  if (DiffLbi4BnInOp("b") != nullptr) {
    const Shape& b_shape = LogicalBlobDesc4BnInOp("b").shape();
    const Shape& out_shape = LogicalBlobDesc4BnInOp("out").shape();
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(b_shape), out_shape.NumAxes());
    if (out_shape == b_shape) {
      *DiffLbi4BnInOp("b") = *DiffLbi4BnInOp("out");
    } else if (out_shape == left_extended_shape) {
      OperatorConf reshape_like_b_op;
      reshape_like_b_op.set_name(op.op_name() + "_grad_b_reshape_like");
      ReshapeLikeOpConf* reshape_like_b_op_conf = reshape_like_b_op.mutable_reshape_like_conf();
      reshape_like_b_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reshape_like_b_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("b")));
      reshape_like_b_op_conf->set_y("y");
      op_confs->push_back(reshape_like_b_op);
      DiffLbi4BnInOp("b")->set_op_name(reshape_like_b_op.name());
      DiffLbi4BnInOp("b")->set_blob_name("y");
    } else {
      OperatorConf reduce_sum_like_b_op;
      reduce_sum_like_b_op.set_name(op.op_name() + "_grad_b");
      ReduceSumLikeOpConf* reduce_sum_like_b_op_conf =
          reduce_sum_like_b_op.mutable_reduce_sum_like_conf();
      reduce_sum_like_b_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      reduce_sum_like_b_op_conf->set_y("y");
      reduce_sum_like_b_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("b")));
      const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(out_shape);
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
