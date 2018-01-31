#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct PoolingCUDACtx {
  int32_t pool_size_h;
  int32_t pool_size_w;
  int32_t strides_h;
  int32_t strides_w;
  int32_t padding_top;
  int32_t padding_bottom;
  int32_t padding_left;
  int32_t padding_right;
};

inline PoolingCUDACtx BuildPoolingCUDACtx(
    const PbMessage& op_conf, const PoolingKernelConf& kernel_conf) {
  PoolingCUDACtx ctx;
  ctx.pool_size_h = GetInt32FromPbMessage(op_conf, "pool_size_h");
  ctx.pool_size_w = GetInt32FromPbMessage(op_conf, "pool_size_w");
  ctx.strides_h = GetInt32FromPbMessage(op_conf, "strides_h");
  ctx.strides_w = GetInt32FromPbMessage(op_conf, "strides_w");
  ctx.padding_top = kernel_conf.padding_top();
  ctx.padding_bottom = kernel_conf.padding_bottom();
  ctx.padding_left = kernel_conf.padding_left();
  ctx.padding_right = kernel_conf.padding_right();
  return ctx;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_
