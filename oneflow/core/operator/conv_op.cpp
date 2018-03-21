#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace {

void GetOutAndPad(const Shape& in_blob_shape, const PbMessage& conv_conf,
                  std::vector<int64_t>* out,
                  std::vector<int32_t>* pad_small_side,
                  std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = GetStringFromPbMessage(conv_conf, "data_format");
  const std::string& padding = GetStringFromPbMessage(conv_conf, "padding");
  const PbRf<int32_t>& dilation_rate =
      GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  const auto& strides = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  const PbRf<int32_t>& kernel_size =
      GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetWindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i),
                          kernel_size.Get(i), dilation_rate.Get(i),
                          strides.Get(i), padding,
                          out ? &(out->at(i)) : nullptr,
                          pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                          pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

}  // namespace

#ifdef WITH_CUDA
CudnnConvDesc::~CudnnConvDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvDesc::CudnnConvDesc(const DataType& data_type,
                             const Shape& in_blob_shape,
                             const PbMessage& conv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(in_blob_shape, conv_conf, nullptr, nullptr, &pad_large_side);
  CudaCheck(cudnnSetConvolutionNdDescriptor(
      val_, opkernel_dim, pad_large_side.data(),
      GetPbRfFromPbMessage<int32_t>(conv_conf, "strides").data(),
      GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate").data(),
      CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
}
#endif  // WITH_CUDA

template<int32_t NDims>
void ConvOp<NDims>::InitFromOpConf() {
  StrFieldTolower("data_format");
  StrFieldTolower("padding");

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
  if (GetBoolFromCustomizedConf("use_bias")) { EnrollModelBn("bias"); }
  EnrollDataTmpBn("cudnn_buf");
}

template<int32_t NDims>
void ConvOp<NDims>::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type) const {
  const std::string& data_format = GetStringFromCustomizedConf("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), NDims + 2);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetInt32FromCustomizedConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }
  std::vector<int64_t> out;
  GetOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr,
               nullptr);
  std::vector<int64_t> out_shape = {data_num, filters};
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + DhwOffset(data_format) + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);

  // weight
  std::vector<int64_t> weight_shape(in_blob_desc->shape().dim_vec());
  weight_shape[0] = filters;
  for (size_t i = 0; i < NDims; ++i) {
    weight_shape[DhwOffset(data_format) + i] =
        GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }
  GetBlobDesc4BnInOp("weight")->mut_shape() = Shape(weight_shape);

  // bias
  if (GetBoolFromCustomizedConf("use_bias")) {
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters});
  }

#ifdef WITH_CUDA
  if (device_type == DeviceType::kGPU) {
    // cudnn_buf
    CudnnConvAlgoCtx conv_ctx;
    InferCudnnAlgo(GetBlobDesc4BnInOp, &conv_ctx);

    int64_t cudnn_buf_size = static_cast<int64_t>(std::max(
        {conv_ctx.fwd_algo_perf.memory, conv_ctx.bwd_filter_algo_perf.memory,
         conv_ctx.bwd_data_algo_perf.memory}));
    GetBlobDesc4BnInOp("cudnn_buf")->mut_shape() = Shape({cudnn_buf_size});
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
void ConvOp<NDims>::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  ConvKernelConf* conv_conf = kernel_conf->mutable_conv_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(conv_conf->mutable_in());
  GetBlobDesc4BnInOp("out")->shape().ToProto(conv_conf->mutable_out());
  GetBlobDesc4BnInOp("weight")->shape().ToProto(conv_conf->mutable_weight());

  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(GetBlobDesc4BnInOp("in")->shape(), GetCustomizedConf(), nullptr,
               &pad_small_side, &pad_large_side);

  for (size_t i = 0; i < NDims; ++i) {
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side",
                                         pad_small_side[i]);
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side",
                                         pad_large_side[i]);
  }

#ifdef WITH_CUDA
  if (kernel_conf->device_type() == DeviceType::kGPU) {
    CudnnConvAlgoCtx conv_ctx;
    InferCudnnAlgo(GetBlobDesc4BnInOp, &conv_ctx);
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_fwd_algo",
        static_cast<int32_t>(conv_ctx.fwd_algo_perf.algo));
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_filter_algo",
        static_cast<int32_t>(conv_ctx.bwd_filter_algo_perf.algo));
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_data_algo",
        static_cast<int32_t>(conv_ctx.bwd_data_algo_perf.algo));
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
PbMessage* ConvOp<NDims>::MutableCustomizedKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_conv_conf();
}

template<int32_t NDims>
int32_t ConvOp<NDims>::ModelSplitAxis() const {
  if (GetStringFromCustomizedConf("data_format") == "channels_first") {
    return 1;
  } else if (GetStringFromCustomizedConf("data_format") == "channels_last") {
    return NDims + 1;
  } else {
    UNIMPLEMENTED();
  }
}

template<int32_t NDims>
int32_t ConvOp<NDims>::MaxModelSplitNum() const {
  return GetInt32FromCustomizedConf("filters");
}

#ifdef WITH_CUDA
template<int32_t NDims>
void ConvOp<NDims>::InferCudnnAlgo(
    std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    CudnnConvAlgoCtx* conv_ctx) const {
  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  DataType data_type = in_blob_desc->data_type();

  std::vector<int32_t> stride_of_in_tensor(NDims + 2, 1);
  std::vector<int32_t> stride_of_out_tensor(NDims + 2, 1);
  for (int32_t i = NDims + 2 - 1; i > 0; --i) {
    for (int32_t j = NDims + 2 - 2; j >= 0; --j) {
      stride_of_in_tensor[j] *= in_blob_desc->shape().At(i);
      stride_of_out_tensor[j] *= out_blob_desc->shape().At(i);
    }
  }
  std::vector<int32_t> in_tensor_dim(in_blob_desc->shape().dim_vec().begin(),
                                     in_blob_desc->shape().dim_vec().end());
  std::vector<int32_t> out_tensor_dim(out_blob_desc->shape().dim_vec().begin(),
                                      out_blob_desc->shape().dim_vec().end());
  CudnnTensorDesc in_desc(data_type, in_tensor_dim.size(), in_tensor_dim.data(),
                          stride_of_in_tensor.data());
  CudnnTensorDesc out_desc(data_type, out_tensor_dim.size(),
                           out_tensor_dim.data(), stride_of_out_tensor.data());
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape(),
                              GetStringFromCustomizedConf("data_format"));
  CudnnConvDesc conv_desc(in_blob_desc->data_type(), in_blob_desc->shape(),
                          GetCustomizedConf());

  int returned_algo_count = -1;
  CudaCheck(cudnnGetConvolutionForwardAlgorithm_v7(
      *cuda_handle.cudnn_handle(), in_desc.Get(), filter_desc.Get(),
      conv_desc.Get(), out_desc.Get(), 1, &returned_algo_count,
      &conv_ctx->fwd_algo_perf));
  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      *cuda_handle.cudnn_handle(), in_desc.Get(), out_desc.Get(),
      conv_desc.Get(), filter_desc.Get(), 1, &returned_algo_count,
      &conv_ctx->bwd_filter_algo_perf));
  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      *cuda_handle.cudnn_handle(), filter_desc.Get(), out_desc.Get(),
      conv_desc.Get(), in_desc.Get(), 1, &returned_algo_count,
      &conv_ctx->bwd_data_algo_perf));
}
#endif  // WITH_CUDA

template class ConvOp<1>;
template class ConvOp<2>;
template class ConvOp<3>;

}  // namespace oneflow
