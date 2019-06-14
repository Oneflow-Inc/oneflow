#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SoftmaxKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernel);
  SoftmaxKernel() = default;
  ~SoftmaxKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct SoftmaxKernelUtil {
  // matrix[i][j] -= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector);

  // matrix[i][j] /= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector);
};

template<DeviceType device_type, typename T>
void SoftmaxComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* tmp,
                        T* prob, void* temp_storage, const size_t temp_storage_bytes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
