#ifndef ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AveragePoolingKernelIf : public PoolingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernelIf);
  AveragePoolingKernelIf() = default;
  virtual ~AveragePoolingKernelIf() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().average_pooling_conf().pooling_conf();
  }
#ifdef WITH_CUDA
  cudnnPoolingMode_t GetCudnnPoolingMode() const override {
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
#endif  // WITH_CUDA
};

template<DeviceType device_type, typename T>
class AveragePoolingKernel final : public AveragePoolingKernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernel);
  AveragePoolingKernel() = default;
  ~AveragePoolingKernel() = default;
};

template<typename T>
class AveragePoolingKernel<DeviceType::kCPU, T> final
    : public AveragePoolingKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernel);
  AveragePoolingKernel() = default;
  virtual ~AveragePoolingKernel() = default;

 private:
  T ForwardInitialize() const override;
  void NCDHWProcess(const T& lhs, T& rhs) const override;
  void NDHWCProcess(const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                    EigenMatrixMap<T>& out_mat) const override;
  void NCDHWFinalize(const int64_t size, T& out) const override;
  void NDHWCFinalize(const int64_t size, const int64_t col,
                     EigenMatrixMap<T>& out_mat) const override;
  void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) const override;
  void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                        ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                        ConstEigenArrayMap<T>& out_diff_arr,
                        EigenArrayMap<T>& in_diff_arr) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_
