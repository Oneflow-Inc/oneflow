#include "oneflow/core/operator/deconv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

void GetDewindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                             const std::string& padding_type, int64_t* output_size,
                             int32_t* padding_before, int32_t* padding_after) {
  CHECK_GT(stride, 0);

  if (padding_type == "valid" || padding_type == "same") {
    if (padding_before) { *padding_before = 0; }
    if (padding_after) { *padding_after = 0; }
    if (output_size) { *output_size = (input_size - 1) * stride + filter_size; }
  } else {
    UNIMPLEMENTED();
  }
  if (output_size) { CHECK_GE((*output_size), 0); }
}

void GetDewindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                             const std::string& padding_type, int64_t* output_size,
                             int32_t* padding_size) {
  GetDewindowedOutputSize(input_size, filter_size, stride, padding_type, output_size, padding_size,
                          nullptr);
}

void GetOutAndPad(const ShapeView& in_blob_shape, const ConvConf& conv_conf, DimVector* out,
                  std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = conv_conf.data_format();
  const std::string& padding = conv_conf.padding();
  const auto& strides = conv_conf.strides();
  const PbRf<int32_t>& kernel_size = conv_conf.kernel_size();
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetDewindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i), kernel_size.Get(i),
                            strides.Get(i), padding, out ? &(out->at(i)) : nullptr,
                            pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                            pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

void GetOutAndPad(const Shape& in_blob_shape, const ConvConf& conv_conf, DimVector* out,
                  std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side) {
  return GetOutAndPad(ShapeView(in_blob_shape), conv_conf, out, pad_small_side, pad_large_side);
}

}  // namespace

#ifdef WITH_CUDA
CudnnDeconvDesc::~CudnnDeconvDesc() { CudaCheck(cudnnDestroyConvolutionDescriptor(val_)); }

CudnnDeconvDesc::CudnnDeconvDesc(const DataType& data_type, const ShapeView& in_blob_shape,
                                 const ConvConf& conv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(in_blob_shape, conv_conf, nullptr, nullptr, &pad_large_side);
  const PbRf<int32_t>& strides = conv_conf.strides();
  const std::vector<int32_t> dilation_rate(opkernel_dim, 1);
  if (opkernel_dim == 2) {
    CudaCheck(cudnnSetConvolution2dDescriptor(
        val_, pad_large_side[0], pad_large_side[1], strides.Get(0), strides.Get(1),
        dilation_rate[0], dilation_rate[1], CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
  } else {
    CudaCheck(cudnnSetConvolutionNdDescriptor(
        val_, opkernel_dim, pad_large_side.data(), strides.data(), dilation_rate.data(),
        CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
  }
}
#endif  // WITH_CUDA

class DeconvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvOp);
  DeconvOp() = default;
  virtual ~DeconvOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().deconv_conf(); }

  void InitFromOpConf() override {
    EnrollInputBn("x");
    EnrollOutputBn("y");
    EnrollInputBn("filter");
    if (op_conf().deconv_conf().use_bias()) { EnrollInputBn("bias"); }
    EnrollTmpBn("cudnn_buf");
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const DeconvOpConf& conf = op_conf().deconv_conf();
    const ConvConf& conv_conf = op_conf().deconv_conf().conv_conf();
    CHECK_OR_RETURN(DevIsGpuAndEnableCudnn()) << "CUDNN is required for Deconv";
    const std::string& data_format = conv_conf.data_format();

    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ_OR_RETURN(x_blob_desc->shape().NumAxes(), NDims() + 2);

    int64_t data_num = x_blob_desc->shape().At(0);
    int32_t filters = conf.filters();
    DimVector out;
    GetOutAndPad(x_blob_desc->shape(), conv_conf, &out, nullptr, nullptr);
    DimVector y_shape = {data_num, filters};
    size_t dhw_offset = DhwOffset(data_format);
    for (size_t i = 0; i < NDims(); ++i) {
      y_shape.insert(y_shape.begin() + dhw_offset + i, out[i]);
    }
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *x_blob_desc;
    y_blob_desc->mut_shape() = Shape(y_shape);

    if (conf.use_bias()) { GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters, 1}); }
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const DeconvOpConf& conf = op_conf().deconv_conf();
    const ConvConf& conv_conf = op_conf().deconv_conf().conv_conf();
    CHECK_OR_RETURN(DevIsGpuAndEnableCudnn()) << "CUDNN is required for Deconv";
    const std::string& data_format = conv_conf.data_format();

    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ_OR_RETURN(x_blob_desc->shape().NumAxes(), NDims() + 2);

    int64_t data_num = x_blob_desc->shape().At(0);
    int64_t channels = x_blob_desc->shape().At(1);
    int32_t filters = conf.filters();
    DimVector out;
    GetOutAndPad(x_blob_desc->shape(), conv_conf, &out, nullptr, nullptr);
    DimVector y_shape = {data_num, filters};
    size_t dhw_offset = DhwOffset(data_format);
    for (size_t i = 0; i < NDims(); ++i) {
      y_shape.insert(y_shape.begin() + dhw_offset + i, out[i]);
    }
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *x_blob_desc;
    y_blob_desc->mut_shape() = Shape(y_shape);

    DimVector weight_shape(y_blob_desc->shape().dim_vec());
    weight_shape[0] = channels;
    if (data_format == "channels_first") {
      weight_shape[1] = filters;
    } else if (data_format == "channels_last") {
      weight_shape[NDims() + 1] = filters;
    } else {
      UNIMPLEMENTED();
    }
    for (size_t i = 0; i < NDims(); ++i) {
      weight_shape[dhw_offset + i] = conv_conf.kernel_size(i);
    }
    BlobDesc* filter_blob_desc = GetBlobDesc4BnInOp("filter");
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("filter")->shape(), Shape(weight_shape));

    if (conf.use_bias()) { GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters, 1}); }

#ifdef WITH_CUDA
    if (DevIsGpuAndEnableCudnn()) {
      DeconvOpCtx* deconv_op_ctx = new DeconvOpCtx();
      EnrollOpCtx(deconv_op_ctx);
      CHECK_OR_RETURN(Global<CudnnConvCtxCache>::Get()->FindCudnnConvAlgoCtxWithConfig(
          *y_blob_desc, *x_blob_desc, *filter_blob_desc, conv_conf, cudnn_buf_limit_byte(),
          this->job_desc().cudnn_conv_enable_true_half(), &deconv_op_ctx->cudnn_deconv_algo_ctx));
      CHECK_OR_RETURN(deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_data_algo_found);
      BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("cudnn_buf");
      cudnn_buf->set_data_type(DataType::kChar);
      size_t buf_size = std::max(size_t(1), deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_data_ws_size);
      cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(buf_size)});
    }
#endif  // WITH_CUDA
    return Maybe<void>::Ok();
  }

 private:
  const int32_t NDims() const { return op_conf().deconv_conf().conv_conf().num_spatial_dims(); }

  PbMessage* MutableCustomizedKernelConf(KernelConf* kernel_conf) const override {
    return kernel_conf->mutable_deconv_conf();
  }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf,
                            const OpContext* op_ctx) const override {
    DeconvKernelConf* deconv_conf = kernel_conf->mutable_deconv_conf();
    deconv_conf->set_dim(NDims());
    GenKernelConfWithCudnn(GetBlobDesc4BnInOp, kernel_conf, deconv_conf, op_ctx);
  }

  void GenKernelConfWithCudnn(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              KernelConf* kernel_conf, DeconvKernelConf* deconv_conf,
                              const OpContext* op_ctx) const {
    GetBlobDesc4BnInOp("x")->shape().ToProto(deconv_conf->mutable_in());
    GetBlobDesc4BnInOp("y")->shape().ToProto(deconv_conf->mutable_out());
    GetBlobDesc4BnInOp("filter")->shape().ToProto(deconv_conf->mutable_weight());

    std::vector<int32_t> pad_small_side;
    std::vector<int32_t> pad_large_side;
    GetOutAndPad(GetBlobDesc4BnInOp("x")->shape(), op_conf().deconv_conf().conv_conf(), nullptr,
                 &pad_small_side, &pad_large_side);

    for (size_t i = 0; i < NDims(); ++i) {
      AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side", pad_small_side[i]);
      AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side", pad_large_side[i]);
    }
#ifdef WITH_CUDA
    if (device_type() == DeviceType::kGPU) {
      const DeconvOpCtx* deconv_op_ctx = static_cast<const DeconvOpCtx*>(op_ctx);
      SetValInCustomizedKernelConf(
          kernel_conf, "cudnn_bwd_data_algo",
          static_cast<int32_t>(deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_data_algo));
    }
#endif  // WITH_CUDA
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("y") = *BatchAxis4BnInOp("x");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("x", 0).Broadcast("filter").Split("y", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kDeconvConf, DeconvOp);

}  // namespace oneflow
