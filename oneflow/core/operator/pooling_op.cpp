#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf() {
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
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void PoolingOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_shape = in_blob_desc->shape();
  CHECK_GE(in_blob_desc->shape().NumAxes(), 3);
  CHECK_LE(in_blob_desc->shape().NumAxes(), 5);
  CHECK_LE(in_blob_desc->shape().NumAxes() - 2, GetDim());
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = GetPoolOutShapeFromInShapeAndPoolConf(
      in_shape, GetDim(), GetValFromCustomizedConf<std::string>("data_format"),
      GetValFromCustomizedConf<std::string>("padding"),
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("pool_size"), GetDim()),
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("strides"), GetDim()));
}

void PoolingOp::CheckPoolSizeAndStrides() const {
  const PbRf<int32_t>& pool_size = GetPbRfFromCustomizedConf<int32_t>("pool_size");
  CHECK_EQ(pool_size.size(), GetDim());
  for (int32_t pool_dim : pool_size) { CHECK_GT(pool_dim, 0); }
  const PbRf<int32_t>& strides = GetPbRfFromCustomizedConf<int32_t>("strides");
  CHECK_EQ(strides.size(), GetDim());
  for (int32_t stride_dim : strides) { CHECK_GT(stride_dim, 0); }
}

Shape GetPoolOutShapeFromInShapeAndPoolConf(const Shape& in_shape, int32_t dim,
                                            const std::string& data_format,
                                            const std::string& padding_type,
                                            const std::vector<int32_t>& pool_size,
                                            const std::vector<int32_t>& strides) {
  std::vector<int64_t> in = {GetInDim(in_shape, data_format, 0, dim),
                             GetInDim(in_shape, data_format, 1, dim),
                             GetInDim(in_shape, data_format, 2, dim)};
  std::vector<int64_t> out;
  Get3DOutputSize(in, pool_size, strides, padding_type, &out, nullptr);

  int32_t n = in_shape.NumAxes();
  CHECK_EQ(n - 2, dim);
  std::vector<int64_t> out_shape(n, 0);
  int32_t c_dim = GetChannelDim(data_format, n);
  out_shape[0] = in_shape.At(0);
  out_shape[c_dim] = in_shape.At(c_dim);
  size_t dhw_offset = DhwOffset(data_format);
  for (int32_t i = 0; i < dim; ++i) { out_shape[i + dhw_offset] = out[i + (3 - dim)]; }
  return Shape(out_shape);
}

void PoolingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  std::string padding_type = GetValFromCustomizedConf<std::string>("padding");
  std::vector<int64_t> in = {GetInDim(in_shape, data_format, 0, GetDim()),
                             GetInDim(in_shape, data_format, 1, GetDim()),
                             GetInDim(in_shape, data_format, 2, GetDim())};
  std::vector<int32_t> pool_size =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("pool_size"), GetDim());
  std::vector<int32_t> strides =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("strides"), GetDim());
  std::vector<int64_t> out;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  Get3DOutputSize(in, pool_size, strides, padding_type, &out, &padding_before, &padding_after);

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
    Shape({in_shape.At(0), in_shape.At(1), in.at(0), in.at(1), in.at(2)})
        .ToProto(pooling_conf->mutable_in());
    Shape({in_shape.At(0), in_shape.At(1), out.at(0), out.at(1), out.at(2)})
        .ToProto(pooling_conf->mutable_out());
  } else if (data_format == "channels_last") {
    Shape({in_shape.At(0), in_shape.At(in_shape.NumAxes() - 1), in.at(0), in.at(1), in.at(2)})
        .ToProto(pooling_conf->mutable_in());
    Shape({in_shape.At(0), in_shape.At(in_shape.NumAxes() - 1), out.at(0), out.at(1), out.at(2)})
        .ToProto(pooling_conf->mutable_out());
  } else {
    UNIMPLEMENTED();
  }
  pooling_conf->set_data_format(data_format);
  pooling_conf->set_padding_type(padding_type);
}

}  // namespace oneflow
