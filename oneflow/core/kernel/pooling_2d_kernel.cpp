#include "oneflow/core/kernel/pooling_2d_kernel.h"

namespace oneflow {

Pooling2DCtx BuildPooling2DCtx(const PbMessage& op_conf,
                               const Pooling2DKernelConf& kernel_conf) {
  Pooling2DCtx ctx;
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
