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
  const Shape& in_shape = in_blob_desc->shape();
  CHECK_GE(in_blob_desc->shape().NumAxes(), 3);
  CHECK_LE(in_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  std::vector<int64_t> in = {GetInDim(in_shape, 0), GetInDim(in_shape, 1),
                             GetInDim(in_shape, 2)};
  std::vector<int64_t> pool_size = GetTensorInOpConf("pool_size");
  std::vector<int64_t> strides = GetTensorInOpConf("strides");
  std::vector<int64_t> out;
  Get3DOutputSize(in, pool_size, strides, GetStringFromSpecialConf("padding"),
                  &out, nullptr);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = GetOutShape(in_shape.At(0), in_shape.At(1), out);
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

void PoolingOp::CheckPoolSizeAndStrides() const {
  const PbRf<int64_t>& pool_size = GetPbRfFromSpecialConf<int64_t>("pool_size");
  CHECK_EQ(pool_size.size(), GetDim());
  for (auto item : pool_size) { CHECK_GT(item, 0); }
  const PbRf<int64_t>& strides = GetPbRfFromSpecialConf<int64_t>("strides");
  CHECK_EQ(strides.size(), GetDim());
  for (auto item : strides) { CHECK_GT(item, 0); }
}

int64_t PoolingOp::GetTensorDimInOpConf(const std::string& field_name,
                                        uint8_t dim) const {
  int64_t index = static_cast<int64_t>(dim) - (3 - GetDim());
  if (index < 0) {
    return 1;
  } else {
    return GetPbRfFromSpecialConf<int64_t>(field_name).Get(index);
  }
}

std::vector<int64_t> PoolingOp::GetTensorInOpConf(
    const std::string& field_name) const {
  std::vector<int64_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    vec.push_back(GetTensorDimInOpConf(field_name, dim));
  }
  return vec;
}

int64_t PoolingOp::GetInDim(const Shape& in_shape, uint8_t dim) const {
  int64_t index = 2 + static_cast<int64_t>(dim) - (3 - GetDim());
  if (index < 2) {
    return 1;
  } else {
    return in_shape.At(index);
  }
}

Shape PoolingOp::GetOutShape(int64_t in_n, int64_t in_c,
                             const std::vector<int64_t>& out) const {
  if (GetDim() == 1) {
    return Shape({in_n, in_c, out.at(2)});
  } else if (GetDim() == 2) {
    return Shape({in_n, in_c, out.at(1), out.at(2)});
  } else if (GetDim() == 3) {
    return Shape({in_n, in_c, out.at(0), out.at(1), out.at(2)});
  } else {
    UNEXPECTED_RUN();
  }
}

void PoolingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  std::vector<int64_t> in = {GetInDim(in_shape, 0), GetInDim(in_shape, 1),
                             GetInDim(in_shape, 2)};
  std::vector<int64_t> pool_size = GetTensorInOpConf("pool_size");
  std::vector<int64_t> strides = GetTensorInOpConf("strides");
  std::vector<int64_t> out;
  std::vector<int64_t> padding;
  Get3DOutputSize(in, pool_size, strides, GetStringFromSpecialConf("padding"),
                  &out, &padding);

  Pooling3DKernelConf* pooling_conf = GetMutPooling3DKernelConf(kernel_conf);
  Shape(pool_size).ToProto(pooling_conf->mutable_pool_size());
  Shape(strides).ToProto(pooling_conf->mutable_strides());
  Shape(padding).ToProto(pooling_conf->mutable_padding());
  Shape({in_shape.At(0), in_shape.At(1), in.at(0), in.at(1), in.at(2)})
      .ToProto(pooling_conf->mutable_in());
  Shape({in_shape.At(0), in_shape.At(1), out.at(0), out.at(1), out.at(2)})
      .ToProto(pooling_conf->mutable_out());
}

}  // namespace oneflow
