#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct Pooling3DCtx {
  int32_t pool_size_d;
  int32_t pool_size_h;
  int32_t pool_size_w;
  int32_t strides_d;
  int32_t strides_h;
  int32_t strides_w;
  int32_t padding_d;
  int32_t padding_h;
  int32_t padding_w;
};

template<DeviceType device_type>
class PoolingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  const Pooling3DCtx& pooling_3d_ctx() const { return pooling_3d_ctx_; }
  Pooling3DCtx* mut_pooling_3d_ctx() { return &pooling_3d_ctx_; }

  Pooling3DCtx pooling_3d_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
