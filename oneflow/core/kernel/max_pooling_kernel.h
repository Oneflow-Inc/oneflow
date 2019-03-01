#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPoolingKernelIf : public PoolingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernelIf);
  MaxPoolingKernelIf() = default;
  virtual ~MaxPoolingKernelIf() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().max_pooling_conf().pooling_conf();
  }
#ifdef WITH_CUDA
  cudnnPoolingMode_t GetCudnnPoolingMode() override { return CUDNN_POOLING_MAX; }
#endif
};

template<DeviceType device_type, typename T>
class MaxPoolingKernel final : public MaxPoolingKernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernel);
  MaxPoolingKernel() = default;
  ~MaxPoolingKernel() = default;
};

template<typename T>
class MaxPoolingKernel<DeviceType::kCPU, T> final : public MaxPoolingKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernel);
  MaxPoolingKernel() = default;
  virtual ~MaxPoolingKernel() = default;

 private:
  T ForwardInitialize() const override;
  void NCDHWProcess(const T& lhs, T& rhs) const override;
  void NDHWCProcess(const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                    EigenMatrixMap<T>& out_mat) const override;
  void NCDHWFinalize(const int64_t size, T& out) const override {}
  void NDHWCFinalize(const int64_t size, const int64_t col,
                     EigenMatrixMap<T>& out_mat) const override {}
  void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) const override;
  void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                        ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                        ConstEigenArrayMap<T>& out_diff_arr,
                        EigenArrayMap<T>& in_diff_arr) const override;
};

template<>
class MaxPoolingKernel<DeviceType::kCPU, float16> final : public Kernel {};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
