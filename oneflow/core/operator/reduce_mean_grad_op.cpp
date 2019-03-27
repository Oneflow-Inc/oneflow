#include "oneflow/core/operator/reduce_mean_grad_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceMeanGradOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_mean_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("x")->set_use_header_only(true);
  EnrollOutputBn("dx");
  EnrollOutputBn("tmp_storage");
}

const PbMessage& ReduceMeanGradOp::GetCustomizedConf() const {
  return op_conf().reduce_mean_grad_conf();
}

void ReduceMeanGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  int64_t y_num_axes = std::max(x_blob_desc->shape().NumAxes(), dy_blob_desc->shape().NumAxes());
  const auto& x_shape = op_conf().reduce_mean_grad_conf().has_kept_dims_shape()
                            ? Shape(op_conf().reduce_mean_grad_conf().kept_dims_shape())
                            : dy_blob_desc->shape().CreateLeftExtendedShape(y_num_axes);
  const auto& like_shape = x_blob_desc->shape().CreateLeftExtendedShape(y_num_axes);
  FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
    CHECK(x_shape.At(i) == 1 || like_shape.At(i) == 1 || x_shape.At(i) == like_shape.At(i));
  }
  *GetBlobDesc4BnInOp("dx") = *x_blob_desc;
}

REGISTER_OP(OperatorConf::kReduceMeanGradConf, ReduceMeanGradOp);

}  // namespace oneflow
