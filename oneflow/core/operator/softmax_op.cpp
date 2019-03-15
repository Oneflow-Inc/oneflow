#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());
  EnrollInputBn("x");
  EnrollOutputBn("y");
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    EnrollOutputBn("transpose_x");
    EnrollOutputBn("transpose_y");
  } else {
    EnrollDataTmpBn("transpose_x");
    EnrollDataTmpBn("transpose_y");
  }
  EnrollFwBufBn("fw_softmax_num");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("transpose_dy");
  EnrollBwBufBn("bw_buf");
  EnrollBwBufBn("bw_softmax_num");
}

const PbMessage& SoftmaxOp::GetCustomizedConf() const { return op_conf().softmax_conf(); }

void SoftmaxOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx, int64_t record_piece_size,
                               std::function<void(OpContext*)> EnrollOpCtx) const {
  // x
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("x");
  // y
  *GetBlobDesc4BnInOp("y") = *in_blob_desc;
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
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_x");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_ctx->axis, in_blob_desc->shape().At(op_ctx->dims - 1));
    transpose_blob_desc->mut_shape().Set(op_ctx->dims - 1, op_ctx->transpose_cols);
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_y") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("transpose_dy") = *transpose_blob_desc;
  }
}

void SoftmaxOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext*, const OpContext* op_ctx) const {
  const SoftmaxOpCtx* softmax_op_ctx = static_cast<const SoftmaxOpCtx*>(op_ctx);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("x");
  // 1D blob store tmp calculate result
  BlobDesc* bw_tmp_blob_desc = GetBlobDesc4BnInOp("bw_softmax_num");
  bw_tmp_blob_desc->mut_shape() = Shape({softmax_op_ctx->transpose_rows});
  bw_tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* bw_buf_blob_desc = GetBlobDesc4BnInOp("bw_buf");
  bw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*in_blob_desc).ByteSizeOfDataContentField())});
  bw_buf_blob_desc->set_data_type(DataType::kChar);
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

SoftmaxOpCtx* SoftmaxOp::NewSoftmaxOpCtx(const Shape& x_shape) const {
  SoftmaxOpCtx* op_ctx = new SoftmaxOpCtx();
  op_ctx->axis = op_conf().softmax_conf().axis();
  op_ctx->dims = x_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->dims, 2);
  CHECK_GE(op_ctx->axis, 1);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  op_ctx->transpose_cols = x_shape.At(op_ctx->axis);
  op_ctx->transpose_rows = x_shape.elem_cnt() / op_ctx->transpose_cols;
  if (op_ctx->axis == op_ctx->dims - 1) {
    op_ctx->need_transpose = false;
  } else {
    op_ctx->need_transpose = true;
  }
  return op_ctx;
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
