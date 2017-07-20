#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class SoftmaxKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernel);
  SoftmaxKernel() = default;
  ~SoftmaxKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename FloatingPointType>
class SoftmaxKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  // n = number of data sample
  // w = number of (input/output) neuron
  static void ForwardMax(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp);

  static void ForwardSum(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp);

  // matrix[i][j] -= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Sub(const KernelCtx& ctx, const int64_t n, const int64_t w,
                  FloatingPointType* matrix, const FloatingPointType* vector);

  static void BackwardDot(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* out,
                          const FloatingPointType* out_diff,
                          FloatingPointType* tmp);
};

template<DeviceType device_type, typename FloatingPointType>
void SoftmaxComputeProb(const KernelCtx& ctx, const int64_t n, const int64_t w,
                        const FloatingPointType* in, FloatingPointType* tmp,
                        FloatingPointType* prob) {
  // copy in blob to prob blob
  KernelUtil<device_type, FloatingPointType>::BlasCopy(ctx, n * w, in, 1, prob,
                                                       1);
  // max | calculate max of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the data of in[i]
  SoftmaxKernelUtil<device_type, FloatingPointType>::ForwardMax(ctx, n, w, prob,
                                                                tmp);
  // sub | every element of prob blob subract the max value of the same sample
  SoftmaxKernelUtil<device_type, FloatingPointType>::Sub(ctx, n, w, prob, tmp);
  // exp | exponentiation every element
  KernelUtil<device_type, FloatingPointType>::Exp(ctx, n * w, prob, prob);
  // sum | calculate sum of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the tmp data after exp
  SoftmaxKernelUtil<device_type, FloatingPointType>::ForwardSum(ctx, n, w, prob,
                                                                tmp);
  // div | every element of prob[i] divided by the data of tmp[i] (the sum
  // value)
  for (int64_t i = 0; i < n; ++i) {
    KernelUtil<device_type, FloatingPointType>::Div(ctx, w, prob + i * w,
                                                    tmp + i);
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
