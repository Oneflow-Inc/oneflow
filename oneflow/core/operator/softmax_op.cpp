#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()
      && op_conf().softmax_conf().axis() != -1) {
    EnrollOutputBn("transpose_in");
    EnrollOutputBn("transpose_out", false);
  } else {
    EnrollDataTmpBn("transpose_in");
    EnrollDataTmpBn("transpose_out");
  }
  EnrollFwBufBn("fw_softmax_num");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("transpose_out_diff");
  EnrollBwBufBn("bw_buf");
  EnrollBwBufBn("bw_softmax_num");
}

const PbMessage& SoftmaxOp::GetCustomizedConf() const { return op_conf().softmax_conf(); }

void SoftmaxOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  op_context_.reset(NewSoftmaxOpCtx(in_blob_desc->shape()));

  // 1D blob store tmp calculate result
  BlobDesc* fw_tmp_blob_desc = GetBlobDesc4BnInOp("fw_softmax_num");
  fw_tmp_blob_desc->mut_shape() = Shape({op_context_->transpose_rows});
  fw_tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
  fw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*in_blob_desc).ByteSizeOfDataContentField())});
  fw_buf_blob_desc->set_data_type(DataType::kChar);
  if (op_context_->need_transpose) {
    // transpose blob
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_in");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_context_->axis,
                                         in_blob_desc->shape().At(op_context_->dims - 1));
    transpose_blob_desc->mut_shape().Set(op_context_->dims - 1, op_context_->transpose_cols);
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_out") = *transpose_blob_desc;
  }
}

void SoftmaxOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext*) const {
  *GetBlobDesc4BnInOp("transpose_out_diff") = *GetBlobDesc4BnInOp("transpose_in");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // 1D blob store tmp calculate result
  BlobDesc* bw_tmp_blob_desc = GetBlobDesc4BnInOp("bw_softmax_num");
  bw_tmp_blob_desc->mut_shape() = Shape({op_context_->transpose_rows});
  bw_tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* bw_buf_blob_desc = GetBlobDesc4BnInOp("bw_buf");
  bw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*in_blob_desc).ByteSizeOfDataContentField())});
  bw_buf_blob_desc->set_data_type(DataType::kChar);
}

void SoftmaxOp::VirtualGenKernelConf(
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

SoftmaxOpCtx* SoftmaxOp::NewSoftmaxOpCtx(const Shape& in_shape) const {
  SoftmaxOpCtx* op_context = new SoftmaxOpCtx();
  op_context->axis = op_conf().softmax_conf().axis();
  op_context->dims = in_shape.NumAxes();
  if (op_context->axis < 0) { op_context->axis += op_context->dims; }
  CHECK_GE(op_context->dims, 2);
  CHECK_GE(op_context->axis, 1);
  CHECK_LT(op_context->axis, op_context->dims);
  op_context->transpose_cols = in_shape.At(op_context->axis);
  op_context->transpose_rows = in_shape.elem_cnt() / op_context->transpose_cols;
  if (op_context->axis == op_context->dims - 1) {
    op_context->need_transpose = false;
  } else {
    op_context->need_transpose = true;
  }
  return op_context;
}

void SoftmaxOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
