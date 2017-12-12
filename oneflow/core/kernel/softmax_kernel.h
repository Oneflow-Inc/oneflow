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
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class SoftmaxKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  // n = number of data sample
  // w = number of (input/output) neuron
  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp);

  static void ForwardSum(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp);

  // matrix[i][j] -= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix,
                  const T* vector);

  static void BackwardDot(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const T* out, const T* out_diff, T* tmp);
};

template<DeviceType device_type, typename T>
void SoftmaxComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w,
                        const T* in, T* tmp, T* prob) {
  // copy in blob to prob blob
  KernelUtil<device_type, T>::Copy(ctx, n * w, in, 1, prob, 1);
  // max | calculate max of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the data of in[i]
  SoftmaxKernelUtil<device_type, T>::ForwardMax(ctx, n, w, prob, tmp);
  // sub | every element of prob blob subract the max value of the same sample
  SoftmaxKernelUtil<device_type, T>::Sub(ctx, n, w, prob, tmp);
  // exp | exponentiation every element
  KernelUtil<device_type, T>::Exp(ctx, n * w, prob, prob);
  // sum | calculate sum of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the tmp data after exp
  SoftmaxKernelUtil<device_type, T>::ForwardSum(ctx, n, w, prob, tmp);
  // div | every element of prob[i] divided by the data of tmp[i] (the sum
  // value)
  for (int64_t i = 0; i < n; ++i) {
    KernelUtil<device_type, T>::Div(ctx, w, prob + i * w, tmp + i);
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
