#ifndef ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type>
class Pooling3DKernel : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernel);
  Pooling3DKernel() = default;
  virtual ~Pooling3DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    const PbMessage& op_conf = GetPooling3DOpConf();
    const PbRpf<int32_t>& pool_size = dynamic_cast<const PbRpf<int32_t>&>(
        (GetMessageFromPbMessage(op_conf, "pool_size")));
    pooling_3d_ctx->pool_size_d = pool_size.Get(0);
    pooling_3d_ctx->pool_size_h = pool_size.Get(1);
    pooling_3d_ctx->pool_size_w = pool_size.Get(2);
    const PbRpf<int32_t>& strides = dynamic_cast<const PbRpf<int32_t>&>(
        (GetMessageFromPbMessage(op_conf, "strides")));
    pooling_3d_ctx->strides_d = strides.Get(0);
    pooling_3d_ctx->strides_h = strides.Get(1);
    pooling_3d_ctx->strides_w = strides.Get(2);

    const Pooling3DKernelConf& kernel_conf = GetPooling3DKernelConf();
    pooling_3d_ctx->padding_d = kernel_conf.padding_d();
    pooling_3d_ctx->padding_h = kernel_conf.padding_h();
    pooling_3d_ctx->padding_w = kernel_conf.padding_w();
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;
  virtual const PbMessage& GetPooling3DOpConf() const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
