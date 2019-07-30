#include "oneflow/core/job/cudnn_conv_ctx_cache_scope.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

CudnnConvCtxCacheScope::CudnnConvCtxCacheScope() {
#ifdef WITH_CUDA
  Global<CudnnConvCtxCache>::New();
#endif
}

CudnnConvCtxCacheScope::~CudnnConvCtxCacheScope() {
#ifdef WITH_CUDA
  Global<CudnnConvCtxCache>::Delete();
#endif
}

}  // namespace oneflow
