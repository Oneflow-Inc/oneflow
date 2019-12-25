#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"

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
    const PbMessage& conf, const size_t max_buf_size, const bool enable_true_half,
    CudnnConvAlgoCtx* conv_algo_ctx) const {
  CudnnConvArgs args(conf, &in_blob_desc, &out_blob_desc, &filter_blob_desc, max_buf_size,
                     GlobalJobDesc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                     GlobalJobDesc().job_conf().cudnn_conv_heuristic_search_algo(),
                     enable_true_half);
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_fwd_algo()) {
    size_t work_space_size;
    const auto algo = static_cast<cudnnConvolutionFwdAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_fwd_algo());
    CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->fwd_algo_found = true;
      conv_algo_ctx->fwd_algo = algo;
      conv_algo_ctx->fwd_ws_size = work_space_size;
    } else {
      conv_algo_ctx->fwd_algo_found = false;
    }
  } else {
    auto fwd_algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionFwdAlgoPerf_t>(args);
    if (fwd_algo_perf->memory <= max_buf_size) {
      conv_algo_ctx->fwd_algo_found = true;
      conv_algo_ctx->fwd_algo = fwd_algo_perf->algo;
      conv_algo_ctx->fwd_ws_size = fwd_algo_perf->memory;
    } else {
      conv_algo_ctx->fwd_algo_found = false;
    }
  }
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_bwd_filter_algo()) {
    size_t work_space_size;
    const auto algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_bwd_filter_algo());
    CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->bwd_filter_algo_found = true;
      conv_algo_ctx->bwd_filter_algo = algo;
      conv_algo_ctx->bwd_filter_ws_size = work_space_size;
    } else {
      conv_algo_ctx->bwd_filter_algo_found = false;
    }
  } else {
    auto bwd_filter_algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>(args);
    if (bwd_filter_algo_perf->memory <= max_buf_size) {
      conv_algo_ctx->bwd_filter_algo_found = true;
      conv_algo_ctx->bwd_filter_algo = bwd_filter_algo_perf->algo;
      conv_algo_ctx->bwd_filter_ws_size = bwd_filter_algo_perf->memory;
    } else {
      conv_algo_ctx->bwd_filter_algo_found = false;
    }
  }
  if (GlobalJobDesc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
    size_t work_space_size;
    const auto algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(
        GlobalJobDesc().job_conf().cudnn_conv_force_bwd_data_algo());
    CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    if (work_space_size <= max_buf_size) {
      conv_algo_ctx->bwd_data_algo_found = true;
      conv_algo_ctx->bwd_data_algo = algo;
      conv_algo_ctx->bwd_data_ws_size = work_space_size;
    } else {
      conv_algo_ctx->bwd_data_algo_found = false;
    }
  } else {
    auto bwd_data_algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>(args);
    if (bwd_data_algo_perf->memory <= max_buf_size) {
      conv_algo_ctx->bwd_data_algo_found = true;
      conv_algo_ctx->bwd_data_algo = bwd_data_algo_perf->algo;
      conv_algo_ctx->bwd_data_ws_size = bwd_data_algo_perf->memory;
    } else {
      conv_algo_ctx->bwd_data_algo_found = false;
    }
  }
  return true;
}

bool CudnnConvCtxCache::FindCudnnConvAlgoCtxWithConfig(
    const BlobDesc& in_blob_desc, const BlobDesc& out_blob_desc, const BlobDesc& filter_blob_desc,
    const PbMessage& conf, const size_t max_buf_size, const bool enable_true_half,
    CudnnConvAlgoCtx* conv_algo_ctx) {
  std::string key = GetKey(in_blob_desc, out_blob_desc, filter_blob_desc, conf, max_buf_size);
  auto algo_ctx_it = conv_config2algo_ctx_.find(key);
  if (algo_ctx_it != conv_config2algo_ctx_.end()) {
    *conv_algo_ctx = algo_ctx_it->second;
    return true;
  } else {
    bool found =
        InferCudnnConvAlgoCtxWithConfig(in_blob_desc, out_blob_desc, filter_blob_desc, conf,
                                        max_buf_size, enable_true_half, conv_algo_ctx);
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

DataType GetConvDescDataType(DataType blob_data_type, const bool enable_true_half) {
  return (blob_data_type == DataType::kFloat16 && !enable_true_half) ? DataType::kFloat
                                                                     : blob_data_type;
}

}  // namespace oneflow

#endif  // WITH_CUDA
