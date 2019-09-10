#ifndef ONEFLOW_CORE_JOB_CUDNN_CONV_CTX_CACHE_SCOPE_H_
#define ONEFLOW_CORE_JOB_CUDNN_CONV_CTX_CACHE_SCOPE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class CudnnConvCtxCacheScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvCtxCacheScope);
  CudnnConvCtxCacheScope();
  ~CudnnConvCtxCacheScope();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CUDNN_CONV_CTX_CACHE_SCOPE_H_
