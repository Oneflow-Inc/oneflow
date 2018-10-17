#include "oneflow/core/operator/deconv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

void GetOutAndPad(const Shape& in_blob_shape, const PbMessage& deconv_conf,
                  std::vector<int64_t>* out, std::vector<int32_t>* pad_small_side,
                  std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = GetValFromPbMessage<std::string>(deconv_conf, "data_format");
  const std::string& padding = GetValFromPbMessage<std::string>(deconv_conf, "padding");
  const auto& strides = GetPbRfFromPbMessage<int32_t>(deconv_conf, "strides");
  const PbRf<int32_t>& kernel_size = GetPbRfFromPbMessage<int32_t>(deconv_conf, "kernel_size");
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetDewindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + 1), kernel_size.Get(i),
                            strides.Get(i), padding, out ? &(out->at(i)) : nullptr,
                            pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                            pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

#ifdef WITH_CUDA
template<typename AlgoPerfType, typename AlgoType>
void FindBestConvAlgo(
    std::function<void(int* num)> GetAlgoMaxCnt,
    std::function<void(int req_num, int* returned_num, AlgoPerfType* results, void* ws)>
        FindAlgoHandler,
    AlgoType* algo, void* work_space) {
  int max_algo_num;
  int returned_algo_num;
  GetAlgoMaxCnt(&max_algo_num);
  AlgoPerfType* perf_results = new AlgoPerfType[max_algo_num];
  FindAlgoHandler(max_algo_num, &returned_algo_num, perf_results, work_space);
  *algo = perf_results[0].algo;
  delete[] perf_results;
}
#endif  // WITH_CUDA

}  // namespace

#ifdef WITH_CUDA
CudnnDeconvDesc::~CudnnDeconvDesc() { CudaCheck(cudnnDestroyConvolutionDescriptor(val_)); }

CudnnDeconvDesc::CudnnDeconvDesc(const DataType& data_type, const Shape& in_blob_shape,
                                 const PbMessage& deconv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(in_blob_shape, deconv_conf, nullptr, nullptr, &pad_large_side);
  const PbRf<int32_t>& strides = GetPbRfFromPbMessage<int32_t>(deconv_conf, "strides");
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

template<int32_t NDims>
void DeconvOp<NDims>::InitFromOpConf() {
  StrFieldTolower("data_format");
  StrFieldTolower("padding");

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
  if (GetValFromCustomizedConf<bool>("use_bias")) { EnrollModelBn("bias"); }
  EnrollDataTmpBn("bw_cudnn_buf");
  EnrollBwBufBn("fw_cudnn_buf");
}

template<int32_t NDims>
void DeconvOp<NDims>::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, std::function<void(OpContext*)> EnrollOpCtx) const {
  CHECK_EQ(parallel_ctx->policy(), ParallelPolicy::kDataParallel)
      << "Deconv only supports data parallel for now";
  CHECK(DevIsGpuAndEnableCudnn()) << "CUDNN is required for Deconv";
  const std::string& data_format = GetValFromCustomizedConf<std::string>("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), NDims + 2);
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int64_t channels = in_blob_desc->shape().At(1);
  int32_t filters = GetValFromCustomizedConf<int32_t>("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(channels, parallel_ctx->parallel_num());
    channels = splitter.At(parallel_ctx->parallel_id()).size();
  }
  std::vector<int64_t> out;
  GetOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr, nullptr);
  std::vector<int64_t> out_shape = {data_num, filters};
  size_t dhw_offset = DhwOffset(data_format);
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + dhw_offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);

  // weight
  std::vector<int64_t> weight_shape(out_blob_desc->shape().dim_vec());
  weight_shape[0] = channels;
  if (data_format == "channels_first") {
    weight_shape[1] = filters;
  } else if (data_format == "channels_last") {
    weight_shape[NDims + 1] = filters;
  } else {
    UNIMPLEMENTED();
  }
  for (size_t i = 0; i < NDims; ++i) {
    weight_shape[dhw_offset + i] = GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }
  GetBlobDesc4BnInOp("weight")->mut_shape() = Shape(weight_shape);

  if (GetValFromCustomizedConf<bool>("use_bias")) {
    //  bias
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters, 1});
  }

  DeconvOpCtx* deconv_op_ctx = new DeconvOpCtx();
  EnrollOpCtx(deconv_op_ctx);

#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) {
    //  cudnn_buf
    InferCudnnAlgo(GetBlobDesc4BnInOp, &(deconv_op_ctx->cudnn_deconv_algo_ctx), 0);
    BlobDesc* bw_cudnn_buf = GetBlobDesc4BnInOp("bw_cudnn_buf");
    bw_cudnn_buf->mut_shape() = Shape(
        {static_cast<int64_t>(std::max(deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_filter_ws_size,
                                       deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_data_ws_size))});
    bw_cudnn_buf->set_data_type(DataType::kChar);
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
void DeconvOp<NDims>::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    const OpContext* op_ctx) const {
  const DeconvOpCtx* deconv_op_ctx = static_cast<const DeconvOpCtx*>(op_ctx);
#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) {
    // cudnn_buf
    BlobDesc* fw_cudnn_buf = GetBlobDesc4BnInOp("fw_cudnn_buf");
    fw_cudnn_buf->mut_shape() =
        Shape({static_cast<int64_t>(deconv_op_ctx->cudnn_deconv_algo_ctx.fwd_ws_size)});
    fw_cudnn_buf->set_data_type(DataType::kChar);
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
void DeconvOp<NDims>::GenKernelConfWithCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, KernelConf* kernel_conf,
    DeconvKernelConf* deconv_conf, const OpContext* op_ctx) const {
  GetBlobDesc4BnInOp("in")->shape().ToProto(deconv_conf->mutable_in());
  GetBlobDesc4BnInOp("out")->shape().ToProto(deconv_conf->mutable_out());
  GetBlobDesc4BnInOp("weight")->shape().ToProto(deconv_conf->mutable_weight());

  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  GetOutAndPad(GetBlobDesc4BnInOp("in")->shape(), GetCustomizedConf(), nullptr, &pad_small_side,
               &pad_large_side);

  for (size_t i = 0; i < NDims; ++i) {
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side", pad_small_side[i]);
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side", pad_large_side[i]);
  }
#ifdef WITH_CUDA
  if (device_type() == DeviceType::kGPU) {
    const DeconvOpCtx* deconv_op_ctx = static_cast<const DeconvOpCtx*>(op_ctx);
    SetValInCustomizedKernelConf(
        kernel_conf, "cudnn_fwd_algo",
        static_cast<int32_t>(deconv_op_ctx->cudnn_deconv_algo_ctx.fwd_algo));
    SetValInCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_filter_algo",
        static_cast<int32_t>(deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_filter_algo));
    SetValInCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_data_algo",
        static_cast<int32_t>(deconv_op_ctx->cudnn_deconv_algo_ctx.bwd_data_algo));
  }
#endif  // WITH_CUDA
}

template<int32_t NDims>
void DeconvOp<NDims>::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  DeconvKernelConf* deconv_conf = kernel_conf->mutable_deconv_conf();
  deconv_conf->set_dim(NDims);
  GenKernelConfWithCudnn(GetBlobDesc4BnInOp, kernel_conf, deconv_conf, op_ctx);
}

template<int32_t NDims>
PbMessage* DeconvOp<NDims>::MutableCustomizedKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_deconv_conf();
}

template<int32_t NDims>
int32_t DeconvOp<NDims>::ModelSplitAxis() const {
  if (GetValFromCustomizedConf<std::string>("data_format") == "channels_first") {
    return 1;
  } else if (GetValFromCustomizedConf<std::string>("data_format") == "channels_last") {
    return NDims + 1;
  } else {
    UNIMPLEMENTED();
  }
}

template<int32_t NDims>
int32_t DeconvOp<NDims>::MaxModelSplitNum() const {
  return GetValFromCustomizedConf<int32_t>("filters");
}

#ifdef WITH_CUDA
template<int32_t NDims>
void DeconvOp<NDims>::InferCudnnAlgo(
    std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    CudnnConvAlgoCtx* deconv_ctx, const int64_t device_id) const {
  CudaStreamHandle cuda_handle(nullptr);

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  std::string format = GetValFromCustomizedConf<std::string>("data_format");

  if (Global<CudnnConvCtxCache>::Get()->FindCudnnConvAlgoCtxWithConfig(
          *out_blob_desc, *in_blob_desc, *weight_blob_desc, format, deconv_ctx)) {
    return;
  }

  DataType data_type = in_blob_desc->data_type();
  CudnnTensorDesc in_desc(data_type, in_blob_desc->shape(), format);
  CudnnTensorDesc out_desc(data_type, out_blob_desc->shape(), format);
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape(), format);
  CudnnDeconvDesc deconv_desc(data_type, in_blob_desc->shape(), GetCustomizedConf());

  size_t avail_ws_sz = cudnn_buf_limit_byte();
  void* in_dptr = nullptr;
  void* out_dptr = nullptr;
  void* filter_dptr = nullptr;
  void* work_space = nullptr;
  CudaCheck(cudaSetDevice(device_id));
  CudaCheck(cudaMalloc(&in_dptr, RtBlobDesc(*in_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&out_dptr, RtBlobDesc(*out_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&filter_dptr, RtBlobDesc(*weight_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&work_space, avail_ws_sz));
  // find best algorithm for forward, which is the backward data function of a convolution
  auto GetBwdDataAlgoMaxCnt = [&](int* num) {
    CudaCheck(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(*cuda_handle.cudnn_handle(), num));
  };
  auto FindBwdDataAlgoHandler = [&](int req_num, int* returned_num,
                                    cudnnConvolutionBwdDataAlgoPerf_t* results, void* ws) {
    CudaCheck(cudnnFindConvolutionBackwardDataAlgorithmEx(
        *cuda_handle.cudnn_handle(), filter_desc.Get(), filter_dptr, in_desc.Get(), in_dptr,
        deconv_desc.Get(), out_desc.Get(), out_dptr, req_num, returned_num, results, ws,
        avail_ws_sz));
  };
  FindBestConvAlgo<cudnnConvolutionBwdDataAlgoPerf_t, decltype(deconv_ctx->bwd_data_algo)>(
      GetBwdDataAlgoMaxCnt, FindBwdDataAlgoHandler, &deconv_ctx->bwd_data_algo, work_space);

  // find best algorithm for backward filter
  auto GetBwdFilterAlgoMaxCnt = [&](int* num) {
    CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(*cuda_handle.cudnn_handle(), num));
  };
  auto FindBwdFilterAlgoHandler = [&](int req_num, int* returned_num,
                                      cudnnConvolutionBwdFilterAlgoPerf_t* results, void* ws) {
    CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        *cuda_handle.cudnn_handle(), out_desc.Get(), out_dptr, in_desc.Get(), in_dptr,
        deconv_desc.Get(), filter_desc.Get(), filter_dptr, req_num, returned_num, results, ws,
        avail_ws_sz));
  };
  FindBestConvAlgo<cudnnConvolutionBwdFilterAlgoPerf_t, decltype(deconv_ctx->bwd_filter_algo)>(
      GetBwdFilterAlgoMaxCnt, FindBwdFilterAlgoHandler, &deconv_ctx->bwd_filter_algo, work_space);

  // find best algorithm for backward data, which is the forward data function of a convolution
  auto GetFwdAlgoMaxCnt = [&](int* num) {
    CudaCheck(cudnnGetConvolutionForwardAlgorithmMaxCount(*cuda_handle.cudnn_handle(), num));
  };
  auto FindFwdAlgoHandler = [&](int req_num, int* returned_num,
                                cudnnConvolutionFwdAlgoPerf_t* results, void* ws) {
    CudaCheck(cudnnFindConvolutionForwardAlgorithmEx(
        *cuda_handle.cudnn_handle(), out_desc.Get(), out_dptr, filter_desc.Get(), filter_dptr,
        deconv_desc.Get(), in_desc.Get(), in_dptr, req_num, returned_num, results, ws,
        avail_ws_sz));
  };
  FindBestConvAlgo<cudnnConvolutionFwdAlgoPerf_t, decltype(deconv_ctx->fwd_algo)>(
      GetFwdAlgoMaxCnt, FindFwdAlgoHandler, &deconv_ctx->fwd_algo, work_space);

  CudaCheck(cudaFree(in_dptr));
  CudaCheck(cudaFree(out_dptr));
  CudaCheck(cudaFree(filter_dptr));
  CudaCheck(cudaFree(work_space));
  in_dptr = nullptr;
  out_dptr = nullptr;
  filter_dptr = nullptr;
  work_space = nullptr;

  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      *cuda_handle.cudnn_handle(), filter_desc.Get(), in_desc.Get(), deconv_desc.Get(),
      out_desc.Get(), deconv_ctx->bwd_data_algo, &deconv_ctx->bwd_data_ws_size));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      *cuda_handle.cudnn_handle(), out_desc.Get(), in_desc.Get(), deconv_desc.Get(),
      filter_desc.Get(), deconv_ctx->bwd_filter_algo, &deconv_ctx->bwd_filter_ws_size));
  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      *cuda_handle.cudnn_handle(), out_desc.Get(), filter_desc.Get(), deconv_desc.Get(),
      in_desc.Get(), deconv_ctx->fwd_algo, &deconv_ctx->fwd_ws_size));

  Global<CudnnConvCtxCache>::Get()->AddCudnnConvAlgoCtxWithConfig(
      *out_blob_desc, *in_blob_desc, *weight_blob_desc, format, *deconv_ctx);
}
#endif  // WITH_CUDA

template class DeconvOp<2>;
template class DeconvOp<3>;

}  // namespace oneflow
