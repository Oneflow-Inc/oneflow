#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace {

void GetOutAndPad(const Shape& in_blob_shape, const std::string& data_format,
                  const std::string& padding, const int32_t* dilation_rate,
                  const int32_t* strides, const int32_t* kernel_size,
                  std::vector<int64_t>& out,
                  std::vector<int32_t>& pad_small_side,
                  std::vector<int32_t>& pad_large_side) {
  int32_t kernel_dim = out.size();
  std::vector<int64_t> in(kernel_dim, 0);

  for (size_t i = 0; i < kernel_dim; ++i) {
    in[i] = in_blob_shape.At(DhwOffset(data_format) + i);
    GetWindowedOutputSize(in[i], kernel_size[i], dilation_rate[i], strides[i],
                          padding, &out[i], &pad_small_side[i],
                          &pad_large_side[i]);
  }
}

}  // namespace

#ifdef WITH_CUDA
CudnnConvDesc::~CudnnConvDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvDesc::CudnnConvDesc(const DataType& data_type,
                             const Shape& in_blob_shape, const int kernel_dim,
                             const int* dilation_rate, const int* strides,
                             const int* kernel_size,
                             const std::string& data_format,
                             const std::string& padding) {
  CHECK_EQ(in_blob_shape.NumAxes(), kernel_dim + 2);
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));

  std::vector<int64_t> out(kernel_dim, 0);
  std::vector<int> pad_small_side(kernel_dim, 0);
  std::vector<int> pad_large_side(kernel_dim, 0);
  GetOutAndPad(in_blob_shape, data_format, padding, dilation_rate, strides,
               kernel_size, out, pad_small_side, pad_large_side);

  CudaCheck(cudnnSetConvolutionNdDescriptor(
      val_, kernel_dim, pad_large_side.data(), strides, dilation_rate,
      CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
}
#endif  // WITH_CUDA

template<int32_t NDims>
void ConvOp<NDims>::InitFromOpConf() {
  CHECK(UseCudnnOnGpu());

  StrFieldTolower("data_format");
  StrFieldTolower("padding");

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
  if (GetBoolFromCustomizedConf("use_bias")) { EnrollModelBn("bias"); }
  if (UseCudnnOnGpu()) { EnrollDataTmpBn("cudnn_workspace"); }
}

template<int32_t NDims>
void ConvOp<NDims>::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const std::string& data_format = GetStringFromCustomizedConf("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), NDims + 2);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetInt32FromCustomizedConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }
  std::vector<int64_t> out(NDims, 0);
  std::vector<int32_t> pad_small_side(NDims, 0);
  std::vector<int32_t> pad_large_side(NDims, 0);
  GetOutAndPad(in_blob_desc->shape(), data_format,
               GetStringFromCustomizedConf("padding"),
               GetPbRfFromCustomizedConf<int32_t>("dilation_rate").data(),
               GetPbRfFromCustomizedConf<int32_t>("strides").data(),
               GetPbRfFromCustomizedConf<int32_t>("kernel_size").data(), out,
               pad_small_side, pad_large_side);

  std::vector<int64_t> out_shape = {data_num, filters};
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + DhwOffset(data_format) + i, out[i]);
  }
  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape(out_shape);
  GetBlobDesc4BnInOp(SoleObn())->set_has_data_id_field(
      in_blob_desc->has_data_id_field());

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
  // cudnn_workspace
  if (UseCudnnOnGpu()) {
    CudnnConvAlgoCtx conv_ctx;
    InferCudnnAlgo(GetBlobDesc4BnInOp, &conv_ctx);

    int64_t cudnn_workspace_size = static_cast<int64_t>(std::max(
        {conv_ctx.fwd_algo_perf.memory, conv_ctx.bwd_filter_algo_perf.memory,
         conv_ctx.bwd_data_algo_perf.memory}));

    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() =
        Shape({cudnn_workspace_size});
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
void ConvOp<NDims>::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto mut_conv_conf = kernel_conf->mutable_conv_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_conv_conf->mutable_in());
  GetBlobDesc4BnInOp("out")->shape().ToProto(mut_conv_conf->mutable_out());
  GetBlobDesc4BnInOp("weight")->shape().ToProto(
      mut_conv_conf->mutable_weight());

  std::vector<int64_t> out(NDims, 0);
  std::vector<int32_t> pad_small_side(NDims, 0);
  std::vector<int32_t> pad_large_side(NDims, 0);
  GetOutAndPad(GetBlobDesc4BnInOp("in")->shape(),
               GetStringFromCustomizedConf("data_format"),
               GetStringFromCustomizedConf("padding"),
               GetPbRfFromCustomizedConf<int32_t>("dilation_rate").data(),
               GetPbRfFromCustomizedConf<int32_t>("strides").data(),
               GetPbRfFromCustomizedConf<int32_t>("kernel_size").data(), out,
               pad_small_side, pad_large_side);

  for (size_t i = 0; i < NDims; ++i) {
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side",
                                         pad_small_side[i]);
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side",
                                         pad_large_side[i]);
  }

#ifdef WITH_CUDA
  if (UseCudnnOnGpu()) {
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
  CudnnConvDesc conv_desc(
      in_blob_desc->data_type(), in_blob_desc->shape(), NDims,
      GetPbRfFromCustomizedConf<int32_t>("dilation_rate").data(),
      GetPbRfFromCustomizedConf<int32_t>("strides").data(),
      GetPbRfFromCustomizedConf<int32_t>("kernel_size").data(),
      GetStringFromCustomizedConf("data_format"),
      GetStringFromCustomizedConf("padding"));

  int returned_algo_count;
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
