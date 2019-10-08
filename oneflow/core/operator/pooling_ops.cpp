#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
namespace oneflow {

class PoolingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  virtual ~PoolingOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual int32_t GetDim() const = 0;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  void CheckPoolSizeAndStrides() const;
  Shape GetOutShape(int64_t in_n, int64_t in_c, const std::vector<int64_t>& out) const;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

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

Maybe<void> PoolingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_shape = in_blob_desc->shape();
  CHECK_GE_OR_RETURN(in_blob_desc->shape().NumAxes(), 3);
  CHECK_LE_OR_RETURN(in_blob_desc->shape().NumAxes(), 5);
  // out
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  std::vector<int64_t> in = {GetInDim(in_shape, data_format, 0, GetDim()),
                             GetInDim(in_shape, data_format, 1, GetDim()),
                             GetInDim(in_shape, data_format, 2, GetDim())};
  std::vector<int32_t> pool_size =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("pool_size"), GetDim());
  std::vector<int32_t> strides =
      Get3DVecInOpConf(GetPbRfFromCustomizedConf<int32_t>("strides"), GetDim());
  std::vector<int64_t> out;
  Get3DOutputSize(in, pool_size, strides, GetValFromCustomizedConf<std::string>("padding"), &out,
                  nullptr);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  int64_t in_c = 0;
  if (data_format == "channels_first") {
    in_c = in_shape.At(1);
  } else if (data_format == "channels_last") {
    in_c = in_shape.At(in_shape.NumAxes() - 1);
  } else {
    UNIMPLEMENTED();
  }
  out_blob_desc->mut_shape() = GetOutShape(in_shape.At(0), in_c, out);
  return Maybe<void>::Ok();
}

void PoolingOp::CheckPoolSizeAndStrides() const {
  const PbRf<int32_t>& pool_size = GetPbRfFromCustomizedConf<int32_t>("pool_size");
  CHECK_EQ(pool_size.size(), GetDim());
  for (int32_t pool_dim : pool_size) { CHECK_GT(pool_dim, 0); }
  const PbRf<int32_t>& strides = GetPbRfFromCustomizedConf<int32_t>("strides");
  CHECK_EQ(strides.size(), GetDim());
  for (int32_t stride_dim : strides) { CHECK_GT(stride_dim, 0); }
}

Shape PoolingOp::GetOutShape(int64_t in_n, int64_t in_c, const std::vector<int64_t>& out) const {
  std::vector<int64_t> out_shape;
  if (GetDim() == 1) {
    out_shape = {out.at(2)};
  } else if (GetDim() == 2) {
    out_shape = {out.at(1), out.at(2)};
  } else if (GetDim() == 3) {
    out_shape = {out.at(0), out.at(1), out.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  if (data_format == "channels_first") {
    out_shape.insert(out_shape.begin(), in_c);
  } else if (data_format == "channels_last") {
    out_shape.insert(out_shape.end(), in_c);
  } else {
    UNIMPLEMENTED();
  }
  out_shape.insert(out_shape.begin(), in_n);
  return Shape(out_shape);
}

void PoolingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
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
  pooling_conf->set_data_format(GetValFromCustomizedConf<std::string>("data_format"));
}

Maybe<void> PoolingOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

template<int32_t NDims>
class PoolingNdOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingNdOp);
  PoolingNdOp() = default;
  virtual ~PoolingNdOp() = default;

 private:
  int32_t GetDim() const override { return NDims; }
};

namespace max_pooling {

class MaxPoolingOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingOp);
  MaxPoolingOp() = default;
  virtual ~MaxPoolingOp() = default;

 private:
  PbMessage* MutableCustomizedKernelConf(KernelConf* kernel_conf) const {
    return kernel_conf->mutable_max_pooling_conf();
  }
};

class MaxPooling1DOp final : public PoolingNdOp<1>, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling1DOp);
  MaxPooling1DOp() = default;
  ~MaxPooling1DOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().max_pooling_1d_conf(); }
};

REGISTER_OP(OperatorConf::kMaxPooling1DConf, MaxPooling1DOp);

class MaxPooling2DOp final : public PoolingNdOp<2>, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling2DOp);
  MaxPooling2DOp() = default;
  ~MaxPooling2DOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().max_pooling_2d_conf(); }
};

REGISTER_OP(OperatorConf::kMaxPooling2DConf, MaxPooling2DOp);

class MaxPooling3DOp final : public PoolingNdOp<3>, public MaxPoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling3DOp);
  MaxPooling3DOp() = default;
  ~MaxPooling3DOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().max_pooling_3d_conf(); }
};

REGISTER_OP(OperatorConf::kMaxPooling3DConf, MaxPooling3DOp);

}  // namespace max_pooling

namespace average_pooling {

class AveragePoolingOp : virtual public PoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingOp);
  AveragePoolingOp() = default;
  virtual ~AveragePoolingOp() = default;

 private:
  PbMessage* MutableCustomizedKernelConf(KernelConf* kernel_conf) const override {
    return kernel_conf->mutable_average_pooling_conf();
  }
};

class AveragePooling1DOp final : public PoolingNdOp<1>, public AveragePoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling1DOp);
  AveragePooling1DOp() = default;
  ~AveragePooling1DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().average_pooling_1d_conf();
  }
};

REGISTER_OP(OperatorConf::kAveragePooling1DConf, AveragePooling1DOp);

class AveragePooling2DOp final : public PoolingNdOp<2>, public AveragePoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling2DOp);
  AveragePooling2DOp() = default;
  ~AveragePooling2DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().average_pooling_2d_conf();
  }
};

REGISTER_OP(OperatorConf::kAveragePooling2DConf, AveragePooling2DOp);

class AveragePooling3DOp final : public PoolingNdOp<3>, public AveragePoolingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DOp);
  AveragePooling3DOp() = default;
  ~AveragePooling3DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().average_pooling_3d_conf();
  }
};

REGISTER_OP(OperatorConf::kAveragePooling3DConf, AveragePooling3DOp);

}  // namespace average_pooling

}  // namespace oneflow
