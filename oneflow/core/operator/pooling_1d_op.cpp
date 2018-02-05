#include "oneflow/core/operator/pooling_1d_op.h"

namespace oneflow {

void Pooling1DOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // NCL
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  int32_t in_length = in_blob_desc->shape().At(2);
  int32_t pool_size_length = GetPoolSizeLength();
  int32_t strides_length = GetStridesLength();
  int32_t padding_length_unused, out_length;
  GetWindowedOutputSize(in_length, pool_size_length, strides_length,
                        GetStringFromSpecialConf("padding"), &out_length,
                        &padding_length_unused);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(
      {in_blob_desc->shape().At(0), in_blob_desc->shape().At(1), out_length});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

void Pooling1DOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int32_t in_length = in_blob_desc->shape().At(2);
  int32_t pool_size_length = GetPoolSizeLength();
  int32_t strides_length = GetStridesLength();
  int32_t padding_length, out_length_unused;
  GetWindowedOutputSize(in_length, pool_size_length, strides_length,
                        GetStringFromSpecialConf("padding"), &out_length_unused,
                        &padding_length);

  Pooling1DKernelConf* pooling_conf = GetMutPooling1DKernelConf(kernel_conf);
  pooling_conf->set_padding_length(padding_length);
}

void Pooling1DOp::VirtualCheckPoolSizeAndStrides() const {
  PbRf<int32_t> pool_size = GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size");
  CHECK_EQ(pool_size.size(), 1);
  for (auto item : pool_size) { CHECK_GT(item, 0); }
  PbRf<int32_t> strides = GetMsgFromSpecialConf<PbRf<int32_t>>("strides");
  CHECK_EQ(strides.size(), 1);
  for (auto item : strides) { CHECK_GT(item, 0); }
}

int32_t Pooling1DOp::GetPoolSizeLength() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size").Get(0);
}

int32_t Pooling1DOp::GetStridesLength() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("strides").Get(0);
}

}  // namespace oneflow
