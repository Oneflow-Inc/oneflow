#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

#ifdef WITH_CUDA

namespace oneflow {
std::string CudnnConvCtxCache::GetKey(const BlobDesc& in_desc, const BlobDesc& out_desc,
                                      const BlobDesc& filter_desc,
                                      const std::string& format) const {
  std::string key = format;
  key += "/" + in_desc.shape().DebugStr() + std::to_string(in_desc.data_type());
  key += "/" + out_desc.shape().DebugStr() + std::to_string(out_desc.data_type());
  key += "/" + filter_desc.shape().DebugStr() + std::to_string(filter_desc.data_type());
  return key;
}

bool CudnnConvCtxCache::FindCudnnConvAlgoCtxWithConfig(const BlobDesc& in_desc,
                                                       const BlobDesc& out_desc,
                                                       const BlobDesc& filter_desc,
                                                       const std::string& format,
                                                       CudnnConvAlgoCtx* conv_algo_ctx) const {
  std::string key = GetKey(in_desc, out_desc, filter_desc, format);
  auto algo_ctx_it = conv_config2algo_ctx_.find(key);
  if (algo_ctx_it != conv_config2algo_ctx_.end()) {
    *conv_algo_ctx = algo_ctx_it->second;
    return true;
  } else {
    return false;
  }
}

void CudnnConvCtxCache::AddCudnnConvAlgoCtxWithConfig(const BlobDesc& in_desc,
                                                      const BlobDesc& out_desc,
                                                      const BlobDesc& filter_desc,
                                                      const std::string& format,
                                                      const CudnnConvAlgoCtx& conv_algo_ctx) {
  std::string key = GetKey(in_desc, out_desc, filter_desc, format);
  CHECK(conv_config2algo_ctx_.emplace(key, conv_algo_ctx).second);
}
}  // namespace oneflow

#endif  // WITH_CUDA
