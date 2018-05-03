#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("softmax_num");
  EnrollDataTmpBn("transpose_in");
  EnrollDataTmpBn("transpose_out");
  EnrollDataTmpBn("transpose_out_diff");
}

const PbMessage& SoftmaxOp::GetCustomizedConf() const { return op_conf().softmax_conf(); }

void SoftmaxOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx,
                               std::function<void(OpContext*)> EnrollOpCtx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  SoftmaxOpCtx* op_ctx = NewSoftmaxOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);

  // 1D blob store tmp calculate result
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("softmax_num");
  tmp_blob_desc->mut_shape() = Shape({op_ctx->transpose_rows});
  tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  if (op_ctx->need_transpose) {
    // transpose blob
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_in");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_ctx->axis, in_blob_desc->shape().At(op_ctx->dims - 1));
    transpose_blob_desc->mut_shape().Set(op_ctx->dims - 1, op_ctx->transpose_cols);
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_out") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("transpose_out_diff") = *transpose_blob_desc;
  }
}

void SoftmaxOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  SoftmaxKernelConf* conf = kernel_conf->mutable_softmax_conf();
  const SoftmaxOpCtx* softmax_ctx = static_cast<const SoftmaxOpCtx*>(op_ctx);
  conf->set_axis(softmax_ctx->axis);
  conf->set_transpose_rows(softmax_ctx->transpose_rows);
  conf->set_transpose_cols(softmax_ctx->transpose_cols);
  conf->set_need_transpose(softmax_ctx->need_transpose);
  if (softmax_ctx->need_transpose) {
    PbRf<int32_t>* perm = conf->mutable_perm();
    perm->Reserve(softmax_ctx->dims);
    for (size_t i = 0; i < softmax_ctx->dims; ++i) { perm->Add(i); }
    (*perm)[softmax_ctx->axis] = softmax_ctx->dims - 1;
    (*perm)[softmax_ctx->dims - 1] = softmax_ctx->axis;
  }
}

SoftmaxOpCtx* SoftmaxOp::NewSoftmaxOpCtx(const Shape& in_shape) const {
  SoftmaxOpCtx* op_ctx = new SoftmaxOpCtx();
  op_ctx->axis = op_conf().softmax_conf().axis();
  op_ctx->dims = in_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->dims, 2);
  CHECK_GE(op_ctx->axis, 1);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  op_ctx->transpose_cols = in_shape.At(op_ctx->axis);
  op_ctx->transpose_rows = in_shape.elem_cnt() / op_ctx->transpose_cols;
  if (op_ctx->axis == op_ctx->dims - 1) {
    op_ctx->need_transpose = false;
  } else {
    op_ctx->need_transpose = true;
  }
  return op_ctx;
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
