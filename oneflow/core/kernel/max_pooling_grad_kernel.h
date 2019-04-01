#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_

#include "oneflow/core/kernel/pooling_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPoolingGradKernelIf : public PoolingGradKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingGradKernelIf);
  MaxPoolingGradKernelIf() = default;
  virtual ~MaxPoolingGradKernelIf() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().max_pooling_conf().pooling_conf();
  }
#ifdef WITH_CUDA
  cudnnPoolingMode_t GetCudnnPoolingMode() override { return CUDNN_POOLING_MAX; }
#endif
};

template<DeviceType device_type, typename T>
class MaxPoolingGradKernel final : public MaxPoolingGradKernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingGradKernel);
  MaxPoolingGradKernel() = default;
  ~MaxPoolingGradKernel() = default;
};

template<typename T>
class MaxPoolingGradKernel<DeviceType::kCPU, T> final
    : public MaxPoolingGradKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingGradKernel);
  MaxPoolingGradKernel() = default;
  virtual ~MaxPoolingGradKernel() = default;

 private:
  void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) const override;
  void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                        ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                        ConstEigenArrayMap<T>& out_diff_arr,
                        EigenArrayMap<T>& in_diff_arr) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
