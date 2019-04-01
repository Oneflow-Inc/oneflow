#include "oneflow/core/operator/pooling_grad_op.h"

namespace oneflow {

void PoolingGradOp::InitFromOpConf() {
  std::string padding_mthd = GetValFromCustomizedConf<std::string>("padding");
  std::transform(padding_mthd.begin(), padding_mthd.end(), padding_mthd.begin(), ::tolower);
  if (padding_mthd != "same" && padding_mthd != "valid") {
    LOG(FATAL) << "Invalid padding method in " << op_name();
  }
  SetValInCustomizedConf("padding", padding_mthd);
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  std::transform(data_format.begin(), data_format.end(), data_format.begin(), ::tolower);
  if (data_format != "channels_last" && data_format != "channels_first") {
    LOG(FATAL) << "Invalid data format in " << op_name();
  }
  SetValInCustomizedConf("data_format", data_format);
  CheckPoolSizeAndStrides();
  EnrollInputBn("x");
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx");
}

void PoolingGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  CHECK_GE(x_blob_desc->shape().NumAxes(), 3);
  CHECK_LE(x_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(x_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  // out
  *GetBlobDesc4BnInOp("dx") = *x_blob_desc;
}

void PoolingGradOp::CheckPoolSizeAndStrides() const {
  const PbRf<int32_t>& pool_size = GetPbRfFromCustomizedConf<int32_t>("pool_size");
  CHECK_EQ(pool_size.size(), GetDim());
  for (int32_t pool_dim : pool_size) { CHECK_GT(pool_dim, 0); }
  const PbRf<int32_t>& strides = GetPbRfFromCustomizedConf<int32_t>("strides");
  CHECK_EQ(strides.size(), GetDim());
  for (int32_t stride_dim : strides) { CHECK_GT(stride_dim, 0); }
}

void PoolingGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& x_shape = GetBlobDesc4BnInOp("x")->shape();
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  std::vector<int64_t> in = {GetInDim(x_shape, data_format, 0, GetDim()),
                             GetInDim(x_shape, data_format, 1, GetDim()),
                             GetInDim(x_shape, data_format, 2, GetDim())};
  std::vector<int32_t> pool_size =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("pool_size"), GetDim());
  std::vector<int32_t> strides =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("strides"), GetDim());
  std::vector<int64_t> out;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  Get3DOutputSize(in, pool_size, strides, GetValFromCustomizedConf<std::string>("padding"), &out,
                  &padding_before, &padding_after);

  auto pooling_conf =
      MutableMsgInCustomizedKernelConf<PoolingKernelConf>(kernel_conf, "pooling_conf");
  pooling_conf->set_dim(GetDim());
  FOR_RANGE(size_t, i, 0, 3) {
    pooling_conf->mutable_pool_size()->Add(pool_size.at(i));
    pooling_conf->mutable_strides()->Add(strides.at(i));
    pooling_conf->mutable_padding_before()->Add(padding_before.at(i));
    pooling_conf->mutable_padding_after()->Add(padding_after.at(i));
  }
  if (data_format == "channels_first") {
    Shape({x_shape.At(0), x_shape.At(1), in.at(0), in.at(1), in.at(2)})
        .ToProto(pooling_conf->mutable_in());
    Shape({x_shape.At(0), x_shape.At(1), out.at(0), out.at(1), out.at(2)})
        .ToProto(pooling_conf->mutable_out());
  } else if (data_format == "channels_last") {
    Shape({x_shape.At(0), x_shape.At(x_shape.NumAxes() - 1), in.at(0), in.at(1), in.at(2)})
        .ToProto(pooling_conf->mutable_in());
    Shape({x_shape.At(0), x_shape.At(x_shape.NumAxes() - 1), out.at(0), out.at(1), out.at(2)})
        .ToProto(pooling_conf->mutable_out());
  } else {
    UNIMPLEMENTED();
  }
  pooling_conf->set_data_format(GetValFromCustomizedConf<std::string>("data_format"));
}

}  // namespace oneflow
