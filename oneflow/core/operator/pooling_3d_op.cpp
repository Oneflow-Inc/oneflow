#include "oneflow/core/operator/pooling_3d_op.h"

namespace oneflow {

void Pooling3DOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // NCDHW
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  int32_t in_d = in_blob_desc->shape().At(2);
  int32_t pool_size_d = GetPoolSizeD();
  int32_t strides_d = GetStridesD();
  int32_t padding_d_unused, out_d;
  GetWindowedOutputSize(in_d, pool_size_d, strides_d,
                        GetStringFromSpecialConf("padding"), &out_d,
                        &padding_d_unused);
  int32_t in_h = in_blob_desc->shape().At(3);
  int32_t pool_size_h = GetPoolSizeH();
  int32_t strides_h = GetStridesH();
  int32_t padding_h_unused, out_h;
  GetWindowedOutputSize(in_h, pool_size_h, strides_h,
                        GetStringFromSpecialConf("padding"), &out_h,
                        &padding_h_unused);
  int32_t in_w = in_blob_desc->shape().At(4);
  int32_t pool_size_w = GetPoolSizeW();
  int32_t strides_w = GetStridesW();
  int32_t padding_w_unused, out_w;
  GetWindowedOutputSize(in_w, pool_size_w, strides_w,
                        GetStringFromSpecialConf("padding"), &out_w,
                        &padding_w_unused);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1), out_d,
             out_h, out_w});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

void Pooling3DOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int32_t in_d = in_blob_desc->shape().At(2);
  int32_t pool_size_d = GetPoolSizeD();
  int32_t strides_d = GetStridesD();
  int32_t padding_d, out_d;
  GetWindowedOutputSize(in_d, pool_size_d, strides_d,
                        GetStringFromSpecialConf("padding"), &out_d,
                        &padding_d);
  int32_t in_h = in_blob_desc->shape().At(3);
  int32_t pool_size_h = GetPoolSizeH();
  int32_t strides_h = GetStridesH();
  int32_t padding_h, out_h;
  GetWindowedOutputSize(in_h, pool_size_h, strides_h,
                        GetStringFromSpecialConf("padding"), &out_h,
                        &padding_h);
  int32_t in_w = in_blob_desc->shape().At(4);
  int32_t pool_size_w = GetPoolSizeW();
  int32_t strides_w = GetStridesW();
  int32_t padding_w, out_w;
  GetWindowedOutputSize(in_w, pool_size_w, strides_w,
                        GetStringFromSpecialConf("padding"), &out_w,
                        &padding_w);

  Pooling3DKernelConf* pooling_conf = GetMutPooling3DKernelConf(kernel_conf);
  pooling_conf->set_padding_d(padding_d);
  pooling_conf->set_padding_h(padding_h);
  pooling_conf->set_padding_w(padding_w);

  pooling_conf->add_in_shape(in_blob_desc->shape().At(0));
  pooling_conf->add_in_shape(in_blob_desc->shape().At(1));
  pooling_conf->add_in_shape(in_blob_desc->shape().At(2));
  pooling_conf->add_in_shape(in_blob_desc->shape().At(3));
  pooling_conf->add_in_shape(in_blob_desc->shape().At(4));

  pooling_conf->add_out_shape(in_blob_desc->shape().At(0));
  pooling_conf->add_out_shape(in_blob_desc->shape().At(1));
  pooling_conf->add_out_shape(out_d);
  pooling_conf->add_out_shape(out_h);
  pooling_conf->add_out_shape(out_w);
}

void Pooling3DOp::VirtualCheckPoolSizeAndStrides() const {
  PbRf<int32_t> pool_size = GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size");
  CHECK_EQ(pool_size.size(), 3);
  for (auto item : pool_size) { CHECK_GT(item, 0); }
  PbRf<int32_t> strides = GetMsgFromSpecialConf<PbRf<int32_t>>("strides");
  CHECK_EQ(strides.size(), 3);
  for (auto item : strides) { CHECK_GT(item, 0); }
}

int32_t Pooling3DOp::GetPoolSizeD() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size").Get(0);
}

int32_t Pooling3DOp::GetPoolSizeH() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size").Get(1);
}

int32_t Pooling3DOp::GetPoolSizeW() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("pool_size").Get(2);
}

int32_t Pooling3DOp::GetStridesD() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("strides").Get(0);
}

int32_t Pooling3DOp::GetStridesH() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("strides").Get(1);
}

int32_t Pooling3DOp::GetStridesW() const {
  return GetMsgFromSpecialConf<PbRf<int32_t>>("strides").Get(2);
}

}  // namespace oneflow
