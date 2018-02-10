#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf() {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  std::transform(padding_mthd.begin(), padding_mthd.end(), padding_mthd.begin(),
                 ::tolower);
  if (padding_mthd != "same" && padding_mthd != "valid") {
    LOG(FATAL) << "Invalid padding method in " << op_name();
  }
  SetStringInSpecialConf("padding", padding_mthd);
  CheckPoolSizeAndStrides();
  EnrollInputBn("in");
  EnrollOutputBn("out");
  VirtualEnrollDataTmpBn();
}

void PoolingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  std::tuple<int32_t, int32_t> res_d =
      CalcOutDAndPaddingD(in_blob_desc->shape());
  std::tuple<int32_t, int32_t> res_h =
      CalcOutHAndPaddingH(in_blob_desc->shape());
  std::tuple<int32_t, int32_t> res_w =
      CalcOutWAndPaddingW(in_blob_desc->shape());

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() =
      CalcOutShape(in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
                   std::get<0>(res_d), std::get<0>(res_h), std::get<0>(res_w));
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

void PoolingOp::CheckPoolSizeAndStrides() const {
  const PbRf<int32_t>& pool_size = GetPbRfFromSpecialConf<int32_t>("pool_size");
  CHECK_EQ(pool_size.size(), GetDim());
  for (auto item : pool_size) { CHECK_GT(item, 0); }
  const PbRf<int32_t>& strides = GetPbRfFromSpecialConf<int32_t>("strides");
  CHECK_EQ(strides.size(), GetDim());
  for (auto item : strides) { CHECK_GT(item, 0); }
}

int32_t PoolingOp::GetDInPbRf(const std::string& field_name) const {
  if (GetDim() == 1) {
    return 1;
  } else if (GetDim() == 2) {
    return 1;
  } else if (GetDim() == 3) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(2);
  } else {
    UNEXPECTED_RUN();
  }
}

int32_t PoolingOp::GetHInPbRf(const std::string& field_name) const {
  if (GetDim() == 1) {
    return 1;
  } else if (GetDim() == 2) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(1);
  } else if (GetDim() == 3) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(2);
  } else {
    UNEXPECTED_RUN();
  }
}

int32_t PoolingOp::GetWInPbRf(const std::string& field_name) const {
  if (GetDim() == 1) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(0);
  } else if (GetDim() == 2) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(1);
  } else if (GetDim() == 3) {
    return GetPbRfFromSpecialConf<int32_t>(field_name).Get(2);
  } else {
    UNEXPECTED_RUN();
  }
}

int32_t PoolingOp::GetInD(const Shape& in_shape) const {
  if (GetDim() == 1) {
    return 1;
  } else if (GetDim() == 2) {
    return 1;
  } else if (GetDim() == 3) {
    return static_cast<int32_t>(in_shape.At(2));
  } else {
    UNEXPECTED_RUN();
  }
}

int32_t PoolingOp::GetInH(const Shape& in_shape) const {
  if (GetDim() == 1) {
    return 1;
  } else if (GetDim() == 2) {
    return static_cast<int32_t>(in_shape.At(2));
  } else if (GetDim() == 3) {
    return static_cast<int32_t>(in_shape.At(3));
  } else {
    UNEXPECTED_RUN();
  }
}

int32_t PoolingOp::GetInW(const Shape& in_shape) const {
  if (GetDim() == 1) {
    return static_cast<int32_t>(in_shape.At(2));
  } else if (GetDim() == 2) {
    return static_cast<int32_t>(in_shape.At(3));
  } else if (GetDim() == 3) {
    return static_cast<int32_t>(in_shape.At(4));
  } else {
    UNEXPECTED_RUN();
  }
}

std::tuple<int32_t, int32_t> PoolingOp::CalcOutDAndPaddingD(
    const Shape& in_shape) const {
  int32_t out_d = 0;
  int32_t padding_d = 0;
  GetWindowedOutputSize(
      GetInD(in_shape), GetDInPbRf("pool_size"), GetDInPbRf("strides"),
      GetStringFromSpecialConf("padding"), &out_d, &padding_d);
  return std::make_tuple(out_d, padding_d);
}

std::tuple<int32_t, int32_t> PoolingOp::CalcOutHAndPaddingH(
    const Shape& in_shape) const {
  int32_t out_h = 0;
  int32_t padding_h = 0;
  GetWindowedOutputSize(
      GetInH(in_shape), GetHInPbRf("pool_size"), GetDInPbRf("strides"),
      GetStringFromSpecialConf("padding"), &out_h, &padding_h);
  return std::make_tuple(out_h, padding_h);
}

std::tuple<int32_t, int32_t> PoolingOp::CalcOutWAndPaddingW(
    const Shape& in_shape) const {
  int32_t out_w = 0;
  int32_t padding_w = 0;
  GetWindowedOutputSize(
      GetInW(in_shape), GetWInPbRf("pool_size"), GetWInPbRf("strides"),
      GetStringFromSpecialConf("padding"), &out_w, &padding_w);
  return std::make_tuple(out_w, padding_w);
}

Shape PoolingOp::CalcOutShape(int32_t in_n, int32_t in_c, int32_t out_d,
                              int32_t out_h, int32_t out_w) const {
  if (GetDim() == 1) {
    return Shape({in_n, in_c, out_w});
  } else if (GetDim() == 2) {
    return Shape({in_n, in_c, out_h, out_w});
  } else if (GetDim() == 3) {
    return Shape({in_n, in_c, out_d, out_h, out_w});
  } else {
    UNEXPECTED_RUN();
  }
}

void PoolingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  std::tuple<int32_t, int32_t> res_d =
      CalcOutDAndPaddingD(in_blob_desc->shape());
  std::tuple<int32_t, int32_t> res_h =
      CalcOutHAndPaddingH(in_blob_desc->shape());
  std::tuple<int32_t, int32_t> res_w =
      CalcOutWAndPaddingW(in_blob_desc->shape());

  Pooling3DKernelConf* pooling_conf = GetMutPooling3DKernelConf(kernel_conf);
  pooling_conf->set_pool_size_d(GetDInPbRf("pool_size"));
  pooling_conf->set_pool_size_h(GetHInPbRf("pool_size"));
  pooling_conf->set_pool_size_w(GetHInPbRf("pool_size"));

  pooling_conf->set_strides_d(GetDInPbRf("strides"));
  pooling_conf->set_strides_h(GetHInPbRf("strides"));
  pooling_conf->set_strides_w(GetWInPbRf("strides"));

  pooling_conf->set_padding_d(std::get<0>(res_d));
  pooling_conf->set_padding_h(std::get<0>(res_h));
  pooling_conf->set_padding_w(std::get<0>(res_w));

  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(0));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(1));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(2));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(3));
  pooling_conf->mutable_in()->add_dim(in_blob_desc->shape().At(4));

  pooling_conf->mutable_out()->add_dim(in_blob_desc->shape().At(0));
  pooling_conf->mutable_out()->add_dim(in_blob_desc->shape().At(1));
  pooling_conf->mutable_out()->add_dim(std::get<1>(res_d));
  pooling_conf->mutable_out()->add_dim(std::get<1>(res_h));
  pooling_conf->mutable_out()->add_dim(std::get<1>(res_w));
}

}  // namespace oneflow
