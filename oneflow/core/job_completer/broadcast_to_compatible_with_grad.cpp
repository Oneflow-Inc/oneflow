#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> GenBroadcastToCompatibleWithGradOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_to_compatible_with_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    const Shape& x_shape = LogicalBlobDesc4BnInOp("x").shape();
    const Shape& y_shape = LogicalBlobDesc4BnInOp("y").shape();
    Shape x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), y_shape.NumAxes());
    std::vector<int32_t> reduced_axes;
    FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
      if (x_extend_shape.At(i) == 1 && y_shape.At(i) != 1) {
        reduced_axes.push_back(i);
      } else {
        CHECK_EQ(x_extend_shape.At(i), y_shape.At(i));
      }
    }
    const auto reduce_sum_like_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-" + op.op_name())
            .Op("reduce_sum_like")
            .Input("x", GenLogicalBlobName(*DiffLbi4BnInOp("y")))
            .Input("like", GenLogicalBlobName(op.BnInOp2Lbi("x")))
            .Attr<std::vector<int32_t>>("axis", reduced_axes)
            .Output("y")
            .Build();
    op_confs->push_back(reduce_sum_like_op.op_conf());
    *DiffLbi4BnInOp("x") = GenLogicalBlobId(reduce_sum_like_op.output("y", 0));
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBroadcastToCompatibleWithConf,
                 &GenBroadcastToCompatibleWithGradOpConf);

}  // namespace oneflow
