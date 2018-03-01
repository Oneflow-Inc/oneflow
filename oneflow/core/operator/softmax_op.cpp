#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

struct SoftmaxOpCtx : public OpContext {
  int32_t axis;
  int32_t transpose_rows;
  int32_t transpose_cols;
  bool need_transpose;
  std::vector<int32_t> perm;
};

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("softmax_num");
  EnrollDataTmpBn("transpose_in");
  EnrollDataTmpBn("transpose_out");
  EnrollDataTmpBn("transpose_out_diff");
}

const PbMessage& SoftmaxOp::GetCustomizedConf() const {
  return op_conf().softmax_conf();
}

void SoftmaxOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpContext) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  int32_t axis = op_conf().softmax_conf().axis();
  int32_t dims = in_blob_desc->shape().NumAxes();
  if (axis < 0) { axis += dims; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, dims);
  Shape in_shape = in_blob_desc->shape();
  int32_t transpose_cols = in_shape.At(axis);
  int32_t transpose_rows = in_shape.elem_cnt() / transpose_cols;
  // 1D blob store tmp calculate result
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("softmax_num");
  tmp_blob_desc->mut_shape() = Shape({transpose_rows});
  tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  SoftmaxOpCtx* op_ctx = new SoftmaxOpCtx;
  op_ctx->axis = axis;
  op_ctx->transpose_rows = transpose_rows;
  op_ctx->transpose_cols = transpose_cols;
  if (axis == dims - 1) {
    op_ctx->need_transpose = false;
  } else {
    // transpose blob
    op_ctx->need_transpose = true;
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_in");
    transpose_blob_desc->mut_shape() = in_shape;
    transpose_blob_desc->mut_shape().Set(axis, in_shape.At(dims - 1));
    transpose_blob_desc->mut_shape().Set(dims - 1, in_shape.At(axis));
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_out") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("transpose_out_diff") = *transpose_blob_desc;
    op_ctx->perm.clear();
    for (size_t i = 0; i < dims; ++i) { op_ctx->perm.push_back(i); }
    op_ctx->perm[axis] = dims - 1;
    op_ctx->perm[dims - 1] = axis;
  }
  EnrollOpContext(op_ctx);
}

void SoftmaxOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const OpContext* op_ctx,
    KernelConf* kernel_conf) const {
  auto conf = kernel_conf->mutable_softmax_conf();
  auto softmax_ctx = static_cast<const SoftmaxOpCtx*>(op_ctx);
  conf->set_axis(softmax_ctx->axis);
  conf->set_transpose_rows(softmax_ctx->transpose_rows);
  conf->set_transpose_cols(softmax_ctx->transpose_cols);
  conf->set_need_transpose(softmax_ctx->need_transpose);
  // *(conf->mutable_perm()) = StdVec2PbRpf(softmax_ctx->perm);
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
