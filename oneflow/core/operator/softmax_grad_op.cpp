#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/operator/softmax_grad_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SoftmaxGradOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_grad_conf());
  EnrollInputBn("y");
  EnrollInputBn("dy");
  if (op_conf().softmax_grad_conf().has_transpose_x()
      && op_conf().softmax_grad_conf().has_transpose_y()) {
    EnrollInputBn("transpose_x");
    EnrollInputBn("transpose_y");
  } else {
    CHECK(!op_conf().softmax_grad_conf().has_transpose_x());
    CHECK(!op_conf().softmax_grad_conf().has_transpose_y());
  }
  EnrollFwBufBn("transpose_dy");
  EnrollFwBufBn("bw_buf");
  EnrollFwBufBn("bw_softmax_num");
  EnrollOutputBn("dx");
}

const PbMessage& SoftmaxGradOp::GetCustomizedConf() const { return op_conf().softmax_grad_conf(); }

void SoftmaxGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  // dy
  const BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
  // dx
  BlobDesc* dx_blob_desc = GetBlobDesc4BnInOp("dx");
  *dx_blob_desc = *dy_blob_desc;
  if (op_conf().softmax_grad_conf().has_transpose_y()) {
    *GetBlobDesc4BnInOp("transpose_dy") = *GetBlobDesc4BnInOp("transpose_y");
  }
  op_context_.reset(NewSoftmaxGradOpCtx(dx_blob_desc->shape()));
  // 1D blob store tmp calculate result
  BlobDesc* bw_tmp_blob_desc = GetBlobDesc4BnInOp("bw_softmax_num");
  bw_tmp_blob_desc->mut_shape() = Shape({op_context_->transpose_rows});
  bw_tmp_blob_desc->set_data_type(dx_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* bw_buf_blob_desc = GetBlobDesc4BnInOp("bw_buf");
  bw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*dx_blob_desc).ByteSizeOfDataContentField())});
  bw_buf_blob_desc->set_data_type(DataType::kChar);
}

void SoftmaxGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  SoftmaxKernelConf* conf = kernel_conf->mutable_softmax_conf();
  conf->set_axis(op_context_->axis);
  conf->set_transpose_rows(op_context_->transpose_rows);
  conf->set_transpose_cols(op_context_->transpose_cols);
  conf->set_need_transpose(op_context_->need_transpose);
  if (op_context_->need_transpose) {
    PbRf<int32_t>* perm = conf->mutable_perm();
    perm->Reserve(op_context_->dims);
    for (size_t i = 0; i < op_context_->dims; ++i) { perm->Add(i); }
    (*perm)[op_context_->axis] = op_context_->dims - 1;
    (*perm)[op_context_->dims - 1] = op_context_->axis;
  }
}

SoftmaxGradOpCtx* SoftmaxGradOp::NewSoftmaxGradOpCtx(const Shape& dx_shape) const {
  SoftmaxGradOpCtx* op_ctx = new SoftmaxGradOpCtx();
  op_ctx->axis = op_conf().softmax_grad_conf().axis();
  op_ctx->dims = dx_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->dims, 2);
  CHECK_GE(op_ctx->axis, 1);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  op_ctx->transpose_cols = dx_shape.At(op_ctx->axis);
  op_ctx->transpose_rows = dx_shape.elem_cnt() / op_ctx->transpose_cols;
  if (op_ctx->axis == op_ctx->dims - 1) {
    op_ctx->need_transpose = false;
  } else {
    op_ctx->need_transpose = true;
  }
  return op_ctx;
}

void SoftmaxGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kSoftmaxGradConf, SoftmaxGradOp);

}  // namespace oneflow
