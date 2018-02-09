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
  int32_t out_d;
  GetWindowedOutputSize(in_d, pool_size_d, strides_d,
                        GetStringFromSpecialConf("padding"), &out_d, nullptr);
  int32_t in_h = in_blob_desc->shape().At(3);
  int32_t pool_size_h = GetPoolSizeH();
  int32_t strides_h = GetStridesH();
  int32_t out_h;
  GetWindowedOutputSize(in_h, pool_size_h, strides_h,
                        GetStringFromSpecialConf("padding"), &out_h, nullptr);
  int32_t in_w = in_blob_desc->shape().At(4);
  int32_t pool_size_w = GetPoolSizeW();
  int32_t strides_w = GetStridesW();
  int32_t out_w;
  GetWindowedOutputSize(in_w, pool_size_w, strides_w,
                        GetStringFromSpecialConf("padding"), &out_w, nullptr);

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
  int32_t padding_d;
  int32_t out_d;
  GetWindowedOutputSize(in_d, pool_size_d, strides_d,
                        GetStringFromSpecialConf("padding"), &out_d,
                        &padding_d);
  int32_t in_h = in_blob_desc->shape().At(3);
  int32_t pool_size_h = GetPoolSizeH();
  int32_t strides_h = GetStridesH();
  int32_t padding_h;
  int32_t out_h;
  GetWindowedOutputSize(in_h, pool_size_h, strides_h,
                        GetStringFromSpecialConf("padding"), &out_h,
                        &padding_h);
  int32_t in_w = in_blob_desc->shape().At(4);
  int32_t pool_size_w = GetPoolSizeW();
  int32_t strides_w = GetStridesW();
  int32_t padding_w;
  int32_t out_w;
  GetWindowedOutputSize(in_w, pool_size_w, strides_w,
                        GetStringFromSpecialConf("padding"), &out_w,
                        &padding_w);

  Pooling3DKernelConf* pooling_conf = GetMutPooling3DKernelConf(kernel_conf);
  pooling_conf->set_pool_size_d(pool_size_d);
  pooling_conf->set_pool_size_h(pool_size_h);
  pooling_conf->set_pool_size_w(pool_size_w);

  pooling_conf->set_strides_d(strides_d);
  pooling_conf->set_strides_h(strides_h);
  pooling_conf->set_strides_w(strides_w);

  pooling_conf->set_padding_d(padding_d);
  pooling_conf->set_padding_h(padding_h);
  pooling_conf->set_padding_w(padding_w);

  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(0));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(1));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(2));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(3));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(4));

  pooling_conf->mutable_out()->add_dim(in_blob_desc->shape().At(0));
  pooling_conf->mutable_out()->add_dim(in_blob_desc->shape().At(1));
  pooling_conf->mutable_out()->add_dim(out_d);
  pooling_conf->mutable_out()->add_dim(out_h);
  pooling_conf->mutable_out()->add_dim(out_w);
}

void Pooling3DOp::VirtualCheckPoolSizeAndStrides() const {
  const PbRf<int32_t>& pool_size = GetPbRfFromSpecialConf<int32_t>("pool_size");
  CHECK_EQ(pool_size.size(), 3);
  for (auto item : pool_size) { CHECK_GT(item, 0); }
  const PbRf<int32_t>& strides = GetPbRfFromSpecialConf<int32_t>("strides");
  CHECK_EQ(strides.size(), 3);
  for (auto item : strides) { CHECK_GT(item, 0); }
}

int32_t Pooling3DOp::GetPoolSizeD() const {
  return GetPbRfFromSpecialConf<int32_t>("pool_size").Get(0);
}

int32_t Pooling3DOp::GetPoolSizeH() const {
  return GetPbRfFromSpecialConf<int32_t>("pool_size").Get(1);
}

int32_t Pooling3DOp::GetPoolSizeW() const {
  return GetPbRfFromSpecialConf<int32_t>("pool_size").Get(2);
}

int32_t Pooling3DOp::GetStridesD() const {
  return GetPbRfFromSpecialConf<int32_t>("strides").Get(0);
}

int32_t Pooling3DOp::GetStridesH() const {
  return GetPbRfFromSpecialConf<int32_t>("strides").Get(1);
}

int32_t Pooling3DOp::GetStridesW() const {
  return GetPbRfFromSpecialConf<int32_t>("strides").Get(2);
}

}  // namespace oneflow
