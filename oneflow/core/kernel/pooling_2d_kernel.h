#ifndef ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct Pooling2DCtx {
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
class Pooling2DKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DKernel);
  Pooling2DKernel() = default;
  virtual ~Pooling2DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    pooling_2d_ctx_ =
        BuildPooling2DCtx(GetPooling2DOpConf(), GetPooling2DKernelConf());
  }
  const Pooling2DCtx& pooling_2d_ctx() const { return pooling_2d_ctx_; }
  virtual const Pooling2DKernelConf& GetPooling2DKernelConf() const = 0;
  virtual const PbMessage& GetPooling2DOpConf() const = 0;

 private:
  Pooling2DCtx pooling_2d_ctx_;
};

Pooling2DCtx BuildPooling2DCtx(const PbMessage& op_conf,
                               const Pooling2DKernelConf& kernel_conf);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
