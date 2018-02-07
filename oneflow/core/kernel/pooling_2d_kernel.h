#ifndef ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type>
class Pooling2DKernel : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DKernel);
  Pooling2DKernel() = default;
  virtual ~Pooling2DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    const PbMessage& op_conf = GetPooling2DOpConf();
    const PbRpf<int32_t>& pool_size = dynamic_cast<const PbRpf<int32_t>&>(
        (GetMessageFromPbMessage(op_conf, "pool_size")));
    pooling_3d_ctx->pool_size_d = 0;
    pooling_3d_ctx->pool_size_h = pool_size.Get(0);
    pooling_3d_ctx->pool_size_w = pool_size.Get(1);
    const PbRpf<int32_t>& strides = dynamic_cast<const PbRpf<int32_t>&>(
        (GetMessageFromPbMessage(op_conf, "strides")));
    pooling_3d_ctx->strides_d = 0;
    pooling_3d_ctx->strides_h = strides.Get(0);
    pooling_3d_ctx->strides_w = strides.Get(1);

    const Pooling2DKernelConf& kernel_conf = GetPooling2DKernelConf();
    pooling_3d_ctx->padding_d = 0;
    pooling_3d_ctx->padding_h = kernel_conf.padding_h();
    pooling_3d_ctx->padding_w = kernel_conf.padding_w();

    pooling_3d_ctx->in_n = kernel_conf.in_shape(0);
    pooling_3d_ctx->in_c = kernel_conf.in_shape(1);
    pooling_3d_ctx->in_d = 1;
    pooling_3d_ctx->in_h = kernel_conf.in_shape(2);
    pooling_3d_ctx->in_w = kernel_conf.in_shape(3);

    pooling_3d_ctx->out_n = kernel_conf.out_shape(0);
    pooling_3d_ctx->out_c = kernel_conf.out_shape(1);
    pooling_3d_ctx->out_d = 1;
    pooling_3d_ctx->out_h = kernel_conf.out_shape(2);
    pooling_3d_ctx->out_w = kernel_conf.out_shape(3);
  }
  virtual const Pooling2DKernelConf& GetPooling2DKernelConf() const = 0;
  virtual const PbMessage& GetPooling2DOpConf() const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
