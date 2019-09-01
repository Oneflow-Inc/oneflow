#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (GlobalJobDesc().IsTrain() && op_conf().softmax_conf().axis() != -1) {
    EnrollOutputBn("transpose_in");
    EnrollOutputBn("transpose_out", false);
  } else {
    EnrollTmpBn("transpose_in");
    EnrollTmpBn("transpose_out");
  }
  EnrollTmpBn("fw_softmax_num");
  EnrollTmpBn("fw_buf");
}

const PbMessage& SoftmaxOp::GetCustomizedConf() const { return op_conf().softmax_conf(); }

Maybe<void> SoftmaxOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  SoftmaxOpCtx* op_ctx = NewSoftmaxOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);

  // 1D blob store tmp calculate result
  BlobDesc* fw_tmp_blob_desc = GetBlobDesc4BnInOp("fw_softmax_num");
  fw_tmp_blob_desc->mut_shape() = Shape({op_ctx->transpose_rows});
  fw_tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
  fw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*in_blob_desc).ByteSizeOfDataContentField())});
  fw_buf_blob_desc->set_data_type(DataType::kChar);
  if (op_ctx->need_transpose) {
    // transpose blob
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_in");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_ctx->axis, in_blob_desc->shape().At(op_ctx->dims - 1));
    transpose_blob_desc->mut_shape().Set(op_ctx->dims - 1, op_ctx->transpose_cols);
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_out") = *transpose_blob_desc;
  }
  return Maybe<void>::Ok();
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

void SoftmaxOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
