#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void GetOutAndPad(const ShapeView& in_blob_shape, const PbMessage& conv_conf, DimVector* out,
                  std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = GetValFromPbMessage<std::string>(conv_conf, "data_format");
  const std::string& padding = GetValFromPbMessage<std::string>(conv_conf, "padding");
  const PbRf<int32_t>& dilation_rate = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  const auto& strides = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  const PbRf<int32_t>& kernel_size = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetWindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i), kernel_size.Get(i),
                          dilation_rate.Get(i), strides.Get(i), padding,
                          out ? &(out->at(i)) : nullptr,
                          pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                          pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

}  // namespace

#ifdef WITH_CUDA
CudnnConvDesc::~CudnnConvDesc() { CudaCheck(cudnnDestroyConvolutionDescriptor(val_)); }

CudnnConvDesc::CudnnConvDesc(const DataType& data_type, const ShapeView& in_blob_shape,
                             const PbMessage& conv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(in_blob_shape, conv_conf, nullptr, nullptr, &pad_large_side);
  const PbRf<int32_t>& strides = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  const PbRf<int32_t>& dilation_rate = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  if (opkernel_dim == 2) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], pad_large_side[1],
                                              strides.Get(0), strides.Get(1), dilation_rate.Get(0),
                                              dilation_rate.Get(1), CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else if (opkernel_dim == 1) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], 0, strides.Get(0), 1,
                                              dilation_rate.Get(0), 1, CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else {
    CudaCheck(cudnnSetConvolutionNdDescriptor(
        val_, opkernel_dim, pad_large_side.data(), strides.data(), dilation_rate.data(),
        CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
  }
}
#endif  // WITH_CUDA

template<int32_t NDims>
void ConvOp<NDims>::InitFromOpConf() {
  StrFieldTolower("data_format");
  StrFieldTolower("padding");

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollInputBn("weight");
  EnrollTmpBn("fw_cudnn_buf");
  EnrollTmpBn("fw_col_buf");
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    CHECK(!GetValFromCustomizedConf<std::string>("bias").empty());
    EnrollInputBn("bias");
    EnrollConstBufBn("bias_multiplier");
  }
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const std::string& data_format = GetValFromCustomizedConf<std::string>("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->shape().NumAxes(), NDims + 2);
  // CHECK_EQ(in_blob_desc->data_type(), job_desc().DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetValFromCustomizedConf<int32_t>("filters");
  // only support data parallel
  CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                  || sbp_signature->bn_in_op2sbp_parallel().at("weight").has_broadcast_parallel());

  DimVector out;
  GetOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr, nullptr);
  DimVector out_shape = {data_num, filters};
  size_t dhw_offset = DhwOffset(data_format);
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + dhw_offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);
  return Maybe<void>::Ok();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const std::string& data_format = GetValFromCustomizedConf<std::string>("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->shape().NumAxes(), NDims + 2);
  // CHECK_EQ(in_blob_desc->data_type(), job_desc().DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetValFromCustomizedConf<int32_t>("filters");

  // only support data parallel
  CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                  || sbp_signature->bn_in_op2sbp_parallel().at("weight").has_broadcast_parallel());

  DimVector out;
  GetOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr, nullptr);
  DimVector out_shape = {data_num, filters};
  size_t dhw_offset = DhwOffset(data_format);
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + dhw_offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);

  // weight
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  DimVector weight_shape(in_blob_desc->shape().dim_vec());
  weight_shape[0] = filters;
  for (size_t i = 0; i < NDims; ++i) {
    weight_shape[dhw_offset + i] = GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }
  CHECK_EQ_OR_RETURN(weight_blob_desc->shape(), Shape(weight_shape));

  if (GetValFromCustomizedConf<bool>("use_bias")) {
    // bias and bias_multiplier
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("bias")->shape(), Shape({filters}));
    if (DevIsGpuAndEnableCudnn() == false) {
      DimVector bias_mul_shape(NDims + 1, 1);
      for (size_t i = 0; i != NDims; ++i) { bias_mul_shape[i + 1] = out_shape[dhw_offset + i]; }
      GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape(bias_mul_shape);
    }
  }

  ConvOpCtx* conv_op_ctx = new ConvOpCtx();
  EnrollOpCtx(conv_op_ctx);

  if (DevIsGpuAndEnableCudnn() == false) {
    // col_buf
    int64_t col_buf_elem_cnt = 1;
    for (size_t i = 0; i != NDims + 1; ++i) { col_buf_elem_cnt *= weight_shape[i + 1]; }
    for (size_t i = 0; i != NDims; ++i) { col_buf_elem_cnt *= out_shape[dhw_offset + i]; }
    conv_op_ctx->col_buf_size = col_buf_elem_cnt * GetSizeOfDataType(in_blob_desc->data_type());
    BlobDesc* fw_col_buf = GetBlobDesc4BnInOp("fw_col_buf");
    fw_col_buf->mut_shape() = Shape({conv_op_ctx->col_buf_size});
    fw_col_buf->set_data_type(DataType::kChar);
  }

#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) {
    // cudnn_buf
    size_t fw_cudnn_buf_size = cudnn_buf_limit_byte();
    if (!out_blob_desc->is_dynamic()) {
      CHECK(Global<CudnnConvCtxCache>::Get()->FindCudnnConvAlgoCtxWithConfig(
          *in_blob_desc, *out_blob_desc, *weight_blob_desc, GetCustomizedConf(),
          static_cast<size_t>(cudnn_buf_limit_byte()),
          this->job_desc().cudnn_conv_enable_true_half(), &conv_op_ctx->cudnn_conv_algo_ctx));
      CHECK(conv_op_ctx->cudnn_conv_algo_ctx.fwd_algo_found)
          << "cudnn fwd algo: " << conv_op_ctx->cudnn_conv_algo_ctx.fwd_algo
          << " algo_workspace_size: " << conv_op_ctx->cudnn_conv_algo_ctx.fwd_ws_size
          << " max_workspace_size: " << fw_cudnn_buf_size;
      fw_cudnn_buf_size = conv_op_ctx->cudnn_conv_algo_ctx.fwd_ws_size;
    }
    fw_cudnn_buf_size = std::max(size_t(1), fw_cudnn_buf_size);
    BlobDesc* fw_cudnn_buf = GetBlobDesc4BnInOp("fw_cudnn_buf");
    fw_cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(fw_cudnn_buf_size)});
    fw_cudnn_buf->set_data_type(DataType::kChar);
  }
#endif  // WITH_CUDA
  return Maybe<void>::Ok();
}

template<int32_t NDims>
void ConvOp<NDims>::GenKernelConfWithoutCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    ConvKernelConf* conv_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  const Shape& weight_shape = GetBlobDesc4BnInOp("weight")->shape();
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  DimVector in = {GetInDim(in_shape, data_format, 0, NDims),
                  GetInDim(in_shape, data_format, 1, NDims),
                  GetInDim(in_shape, data_format, 2, NDims)};
  DimVector out;
  std::vector<int32_t> weight =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("kernel_size"), NDims);
  std::vector<int32_t> strides =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("strides"), NDims);
  std::vector<int32_t> dilation_rate =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("dilation_rate"), NDims);
  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  Get3DOutputSize(in, weight, strides, GetValFromCustomizedConf<std::string>("padding"), &out,
                  &pad_small_side, &pad_large_side, &dilation_rate);
  FOR_RANGE(size_t, i, 0, 3) {
    conv_conf->mutable_strides()->Add(strides.at(i));
    conv_conf->mutable_pad_small_side()->Add(pad_small_side.at(i));
    conv_conf->mutable_pad_large_side()->Add(pad_large_side.at(i));
    conv_conf->mutable_dilation_rate()->Add(dilation_rate.at(i));
  }
  const Shape& out_shape = GetBlobDesc4BnInOp("out")->shape();
  if (data_format == "channels_first") {
    Shape({in_shape.At(0), in_shape.At(1), in.at(0), in.at(1), in.at(2)})
        .ToProto(conv_conf->mutable_in());
    Shape({out_shape.At(0), out_shape.At(1), out.at(0), out.at(1), out.at(2)})
        .ToProto(conv_conf->mutable_out());
    Shape({weight_shape.At(0), weight_shape.At(1), weight.at(0), weight.at(1), weight.at(2)})
        .ToProto(conv_conf->mutable_weight());
  } else if (data_format == "channels_last") {
    Shape({in_shape.At(0), in.at(0), in.at(1), in.at(2), in_shape.At(in_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_in());
    Shape({out_shape.At(0), out.at(0), out.at(1), out.at(2), out_shape.At(out_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_out());
    Shape({weight_shape.At(0), weight.at(0), weight.at(1), weight.at(2),
           weight_shape.At(weight_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_weight());
  } else {
    UNIMPLEMENTED();
  }
}

template<int32_t NDims>
void ConvOp<NDims>::GenKernelConfWithCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, KernelConf* kernel_conf,
    ConvKernelConf* conv_conf, const OpContext* op_ctx) const {
  GetBlobDesc4BnInOp("in")->shape().ToProto(conv_conf->mutable_in());
  GetBlobDesc4BnInOp("out")->shape().ToProto(conv_conf->mutable_out());
  GetBlobDesc4BnInOp("weight")->shape().ToProto(conv_conf->mutable_weight());
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    GetBlobDesc4BnInOp("bias")->shape().ToProto(conv_conf->mutable_bias());
  }

  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(GetBlobDesc4BnInOp("in")->shape(), GetCustomizedConf(), nullptr, &pad_small_side,
               &pad_large_side);

  for (size_t i = 0; i < NDims; ++i) {
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side", pad_small_side[i]);
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side", pad_large_side[i]);
  }
}

template<int32_t NDims>
void ConvOp<NDims>::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  ConvKernelConf* conv_conf = kernel_conf->mutable_conv_conf();
  conv_conf->set_dim(NDims);
  if (DevIsGpuAndEnableCudnn()) {
    GenKernelConfWithCudnn(GetBlobDesc4BnInOp, kernel_conf, conv_conf, op_ctx);
  } else {
    GenKernelConfWithoutCudnn(GetBlobDesc4BnInOp, conv_conf);
  }
}

template<int32_t NDims>
PbMessage* ConvOp<NDims>::MutableCustomizedKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_conv_conf();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  return Maybe<void>::Ok();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    SbpSignatureBuilder()
        .Split("in", 0)
        .Broadcast({"weight", "bias"})
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    SbpSignatureBuilder().Split("in", 0).Broadcast("weight").Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

template class ConvOp<1>;
template class ConvOp<2>;
template class ConvOp<3>;

}  // namespace oneflow
