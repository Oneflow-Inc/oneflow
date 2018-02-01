#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct PoolingCtx {
  int32_t pool_size_h;
  int32_t pool_size_w;
  int32_t strides_h;
  int32_t strides_w;
  int32_t padding_top;
  int32_t padding_bottom;
  int32_t padding_left;
  int32_t padding_right;
};

template<DeviceType device_type>
class PoolingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    pooling_ctx_ = BuildPoolingCtx(GetPoolingOpConf(), GetPoolingKernelConf());
  }
  const PoolingCtx& pooling_ctx() const { return pooling_ctx_; }
  virtual const PoolingKernelConf& GetPoolingKernelConf() const = 0;
  virtual const PbMessage& GetPoolingOpConf() const = 0;

 private:
  PoolingCtx pooling_ctx_;
};

PoolingCtx BuildPoolingCtx(const PbMessage& op_conf,
                           const PoolingKernelConf& kernel_conf);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
