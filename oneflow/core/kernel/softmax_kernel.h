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
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct SoftmaxKernelUtil {
  // n = number of data sample
  // w = number of (input/output) neuron
  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* out, T* tmp);

  static void RowSum(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* matrix, T* sum_vec,
                     const T* sum_multiplier);

  // matrix[i][j] -= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector);

  // matrix[i][j] /= vector[i]
  // matrix shape = n*w, vector shape = n
  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector);
};

template<DeviceType device_type, typename T>
void SoftmaxComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* tmp,
                        T* prob, const T* sum_multiplier) {
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
  SoftmaxKernelUtil<device_type, T>::RowSum(ctx, n, w, prob, tmp, sum_multiplier);
  // div | every element of prob[i] divided by the data of tmp[i] (the sum
  // value)
  SoftmaxKernelUtil<device_type, T>::Div(ctx, n, w, prob, tmp);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
