#ifndef ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class Pooling3DKernel : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernel);
  Pooling3DKernel() = default;
  virtual ~Pooling3DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    const Pooling3DKernelConf& kernel_conf = GetPooling3DKernelConf();

    pooling_3d_ctx->InitFromKernelConf(kernel_conf);
    pooling_3d_ctx->BuildCudnnDescs(this->GetPoolingMode(),
                                    GetDataType<T>::val);
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
