#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct PoolingCudaCtx {
  int32_t pool_size_h;
  int32_t pool_size_w;
  int32_t strides_h;
  int32_t strides_w;
  int32_t padding_top;
  int32_t padding_bottom;
  int32_t padding_left;
  int32_t padding_right;
};

PoolingCudaCtx BuildPoolingCudaCtx(const PbMessage& op_conf,
                                   const PoolingKernelConf& kernel_conf);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_
