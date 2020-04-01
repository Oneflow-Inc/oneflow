#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

namespace {

void GenBroadcastToCompatibleWithGradOpConf(
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

    OperatorConf reduce_sum_like_op;
    reduce_sum_like_op.set_name("System-AutoGrad-" + op.op_name());
    ReduceSumLikeOpConf* conf = reduce_sum_like_op.mutable_reduce_sum_like_conf();
    conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("y")));
    conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("x")));
    conf->set_y("y");
    *conf->mutable_axis() = StdVec2PbRf(reduced_axes);
    op_confs->push_back(reduce_sum_like_op);
    DiffLbi4BnInOp("x")->set_op_name(reduce_sum_like_op.name());
    DiffLbi4BnInOp("x")->set_blob_name(conf->y());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBroadcastToCompatibleWithConf,
                 &GenBroadcastToCompatibleWithGradOpConf);

}  // namespace oneflow
