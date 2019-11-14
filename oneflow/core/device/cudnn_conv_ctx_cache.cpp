#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/device/cuda_util.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace {

template<typename T>
std::string FormatPbRf(const PbRf<T>& rf) {
  std::stringstream ss;
  ss << "[";
  for (auto it = rf.cbegin(); it != rf.cend(); ++it) {
    ss << *it;
    if (it != rf.cend() - 1) { ss << ","; }
  }
  ss << "]";
  return ss.str();
}

}  // namespace

std::string CudnnConvCtxCache::GetKey(const BlobDesc& in_blob_desc, const BlobDesc& out_blob_desc,
                                      const BlobDesc& filter_blob_desc, const PbMessage& conf,
                                      const size_t max_buf_size) const {
  std::stringstream key;
  key << GetValFromPbMessage<std::string>(conf, "data_format");
  key << "/" << max_buf_size;
  key << "/" << FormatPbRf<int32_t>(GetPbRfFromPbMessage<int32_t>(conf, "kernel_size"));
  key << "/" << GetValFromPbMessage<std::string>(conf, "padding");
  key << "/" << FormatPbRf<int32_t>(GetPbRfFromPbMessage<int32_t>(conf, "strides"));
  key << "/" << FormatPbRf<int32_t>(GetPbRfFromPbMessage<int32_t>(conf, "dilation_rate"));
  key << "/" << in_blob_desc.shape().DebugStr() + std::to_string(in_blob_desc.data_type());
  key << "/" << out_blob_desc.shape().DebugStr() + std::to_string(out_blob_desc.data_type());
  key << "/" << filter_blob_desc.shape().DebugStr() + std::to_string(filter_blob_desc.data_type());
  return key.str();
}

bool CudnnConvCtxCache::InferCudnnConvAlgoCtxWithConfig(
    const BlobDesc& in_blob_desc, const BlobDesc& out_blob_desc, const BlobDesc& filter_blob_desc,
    const PbMessage& conf, const size_t max_buf_size, CudnnConvAlgoCtx* conv_algo_ctx) const {
  const std::string format = GetValFromPbMessage<std::string>(conf, "data_format");
  const DataType data_type = in_blob_desc.data_type();
  CudnnTensorDesc in_desc(data_type, in_blob_desc.shape(), format);
  CudnnTensorDesc out_desc(data_type, out_blob_desc.shape(), format);
  CudnnFilterDesc filter_desc(data_type, filter_blob_desc.shape(), format);
  CudnnConvDesc conv_desc(GetConvDescDataType(data_type), in_blob_desc.shape(), conf);
  cudnnHandle_t cudnn_handle;
  if (IsCuda9OnTuringDevice()) {
    CudaCheck(cudaDeviceSynchronize());
    CudaCheck(cudaGetLastError());
  }
  CudaCheck(cudnnCreate(&cudnn_handle));
  if (IsCuda9OnTuringDevice()) {
    CudaCheck(cudaDeviceSynchronize());
    cudaGetLastError();
  }
  void* in_dptr = nullptr;
  void* out_dptr = nullptr;
  void* filter_dptr = nullptr;
  void* work_space = nullptr;
  CudaCheck(cudaMalloc(&in_dptr, RtBlobDesc(in_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&filter_dptr, RtBlobDesc(filter_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&out_dptr, RtBlobDesc(out_blob_desc).ByteSizeOfBlobBody()));
  CudaCheck(cudaMalloc(&work_space, max_buf_size));
  int32_t algo_max_cnt;
  int32_t found_algo_cnt;
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_fwd_algo()) {
    size_t work_space_size;
    const cudnnConvolutionFwdAlgo_t algo = static_cast<cudnnConvolutionFwdAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_fwd_algo());
    CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, in_desc.Get(),
                                                      filter_desc.Get(), conv_desc.Get(),
                                                      out_desc.Get(), algo, &work_space_size));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->fwd_algo_found = true;
      conv_algo_ctx->fwd_algo = algo;
      conv_algo_ctx->fwd_ws_size = work_space_size;
    } else {
      conv_algo_ctx->fwd_algo_found = false;
    }
  } else {
    CudaCheck(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &algo_max_cnt));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_vec(static_cast<size_t>(algo_max_cnt));
    CudaCheck(cudnnFindConvolutionForwardAlgorithmEx(
        cudnn_handle, in_desc.Get(), in_dptr, filter_desc.Get(), filter_dptr, conv_desc.Get(),
        out_desc.Get(), out_dptr, algo_max_cnt, &found_algo_cnt, fwd_algo_perf_vec.data(),
        work_space, max_buf_size));
    if (found_algo_cnt > 0) {
      conv_algo_ctx->fwd_algo_found = true;
      conv_algo_ctx->fwd_algo = fwd_algo_perf_vec.at(0).algo;
      conv_algo_ctx->fwd_ws_size = fwd_algo_perf_vec.at(0).memory;
    } else {
      conv_algo_ctx->fwd_algo_found = false;
    }
  }
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_bwd_filter_algo()) {
    size_t work_space_size;
    const cudnnConvolutionBwdFilterAlgo_t algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_bwd_filter_algo());
    CudaCheck((cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_handle, in_desc.Get(), out_desc.Get(), conv_desc.Get(), filter_desc.Get(), algo,
        &work_space_size)));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->bwd_filter_algo_found = true;
      conv_algo_ctx->bwd_filter_algo = algo;
      conv_algo_ctx->bwd_filter_ws_size = work_space_size;
    } else {
      conv_algo_ctx->bwd_filter_algo_found = false;
    }
  } else {
    CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &algo_max_cnt));
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_vec(
        static_cast<size_t>(algo_max_cnt));
    CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        cudnn_handle, in_desc.Get(), in_dptr, out_desc.Get(), out_dptr, conv_desc.Get(),
        filter_desc.Get(), filter_dptr, algo_max_cnt, &found_algo_cnt,
        bwd_filter_algo_perf_vec.data(), work_space, max_buf_size));
    if (found_algo_cnt > 0) {
      conv_algo_ctx->bwd_filter_algo_found = true;
      conv_algo_ctx->bwd_filter_algo = bwd_filter_algo_perf_vec.at(0).algo;
      conv_algo_ctx->bwd_filter_ws_size = bwd_filter_algo_perf_vec.at(0).memory;
    } else {
      conv_algo_ctx->bwd_filter_algo_found = false;
    }
  }
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
    size_t work_space_size;
    const cudnnConvolutionBwdDataAlgo_t algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_bwd_data_algo());
    CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, filter_desc.Get(),
                                                           out_desc.Get(), conv_desc.Get(),
                                                           in_desc.Get(), algo, &work_space_size));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->bwd_data_algo_found = true;
      conv_algo_ctx->bwd_data_algo = algo;
      conv_algo_ctx->bwd_data_ws_size = work_space_size;
    } else {
      conv_algo_ctx->bwd_data_algo_found = false;
    }
  } else {
    CudaCheck(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle, &algo_max_cnt));
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_perf_vec(
        static_cast<size_t>(algo_max_cnt));
    CudaCheck(cudnnFindConvolutionBackwardDataAlgorithmEx(
        cudnn_handle, filter_desc.Get(), filter_dptr, out_desc.Get(), out_dptr, conv_desc.Get(),
        in_desc.Get(), in_dptr, algo_max_cnt, &found_algo_cnt, bwd_data_algo_perf_vec.data(),
        work_space, max_buf_size));
    if (found_algo_cnt > 0) {
      conv_algo_ctx->bwd_data_algo_found = true;
      conv_algo_ctx->bwd_data_algo = bwd_data_algo_perf_vec.at(0).algo;
      conv_algo_ctx->bwd_data_ws_size = bwd_data_algo_perf_vec.at(0).memory;
    } else {
      conv_algo_ctx->bwd_data_algo_found = false;
    }
  }
  CudaCheck(cudaFree(in_dptr));
  CudaCheck(cudaFree(out_dptr));
  CudaCheck(cudaFree(filter_dptr));
  CudaCheck(cudaFree(work_space));
  in_dptr = nullptr;
  out_dptr = nullptr;
  filter_dptr = nullptr;
  work_space = nullptr;
  CudaCheck(cudnnDestroy(cudnn_handle));
  return true;
}

bool CudnnConvCtxCache::FindCudnnConvAlgoCtxWithConfig(
    const BlobDesc& in_blob_desc, const BlobDesc& out_blob_desc, const BlobDesc& filter_blob_desc,
    const PbMessage& conf, const size_t max_buf_size, CudnnConvAlgoCtx* conv_algo_ctx) {
  std::string key = GetKey(in_blob_desc, out_blob_desc, filter_blob_desc, conf, max_buf_size);
  auto algo_ctx_it = conv_config2algo_ctx_.find(key);
  if (algo_ctx_it != conv_config2algo_ctx_.end()) {
    *conv_algo_ctx = algo_ctx_it->second;
    return true;
  } else {
    bool found = InferCudnnConvAlgoCtxWithConfig(in_blob_desc, out_blob_desc, filter_blob_desc,
                                                 conf, max_buf_size, conv_algo_ctx);
    if (found) {
      AddCudnnConvAlgoCtxWithConfig(in_blob_desc, out_blob_desc, filter_blob_desc, conf,
                                    max_buf_size, *conv_algo_ctx);
    }
    return found;
  }
}

void CudnnConvCtxCache::AddCudnnConvAlgoCtxWithConfig(
    const BlobDesc& in_blob_desc, const BlobDesc& out_blob_desc, const BlobDesc& filter_blob_desc,
    const PbMessage& conf, const size_t max_buf_size, const CudnnConvAlgoCtx& conv_algo_ctx) {
  std::string key = GetKey(in_blob_desc, out_blob_desc, filter_blob_desc, conf, max_buf_size);
  CHECK(conv_config2algo_ctx_.emplace(key, conv_algo_ctx).second);
}

DataType GetConvDescDataType(DataType blob_data_type) {
  DataType conv_desc_data_type =
      blob_data_type == DataType::kFloat16 && !GlobalJobDesc().enable_true_half_config_when_conv()
          ? DataType::kFloat
          : blob_data_type;
  return conv_desc_data_type;
}

}  // namespace oneflow

#endif  // WITH_CUDA
