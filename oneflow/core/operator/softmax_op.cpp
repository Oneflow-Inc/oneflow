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

const PbMessage& SoftmaxOp::GetCustomizedConf() const {
  return op_conf().softmax_conf();
}

void SoftmaxOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  Shape in_shape = in_blob_desc->shape();
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  int32_t axis = op_conf().softmax_conf().axis();
  int32_t dims = in_shape.NumAxes();
  if (axis < 0) { axis += dims; }
  CHECK_GE(dims, 2);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, dims);
  // 1D blob store tmp calculate result
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("softmax_num");
  tmp_blob_desc->mut_shape() = Shape({in_shape.elem_cnt() / in_shape.At(axis)});
  tmp_blob_desc->set_data_type(in_blob_desc->data_type());
  if (axis != dims - 1) {
    // transpose blob
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("transpose_in");
    transpose_blob_desc->mut_shape() = in_shape;
    transpose_blob_desc->mut_shape().Set(axis, in_shape.At(dims - 1));
    transpose_blob_desc->mut_shape().Set(dims - 1, in_shape.At(axis));
    transpose_blob_desc->set_data_type(in_blob_desc->data_type());
    *GetBlobDesc4BnInOp("transpose_out") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("transpose_out_diff") = *transpose_blob_desc;
  }
}

void SoftmaxOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto conf = kernel_conf->mutable_softmax_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  Shape in_shape = in_blob_desc->shape();
  int32_t axis = op_conf().softmax_conf().axis();
  int32_t dims = in_blob_desc->shape().NumAxes();
  if (axis < 0) { axis += dims; }
  int64_t transpose_cols = in_shape.At(axis);
  int64_t transpose_rows = in_shape.elem_cnt() / transpose_cols;
  conf->set_axis(axis);
  conf->set_transpose_rows(transpose_rows);
  conf->set_transpose_cols(transpose_cols);
  if (axis == dims - 1) {
    conf->set_need_transpose(false);
  } else {
    conf->set_need_transpose(true);
    PbRf<int32_t>* perm = conf->mutable_perm();
    perm->Reserve(dims);
    for (size_t i = 0; i < dims; ++i) { perm->Add(i); }
    (*perm)[axis] = dims - 1;
    (*perm)[dims - 1] = axis;
  }
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
