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
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename U = void>
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
                        T* prob, void* temp_storage, const size_t temp_storage_bytes) {
  // copy in blob to prob blob
  // KernelUtil<device_type, T>::Copy(ctx, n * w, in, 1, prob, 1);
  Memcpy<device_type>(ctx, static_cast<void*>(prob), static_cast<const void*>(in),
                      static_cast<size_t>(n * w * sizeof(T)));
  // max | calculate max of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the data of in[i]
  /* ndarray reduce max
  NdarrayUtil<device_type, T>::ReduceMax(
      ctx, XpuVarNdarray<T>({n, 1}, tmp), XpuVarNdarray<const T>({n, w}, prob),
      XpuVarNdarray<T>({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                       reinterpret_cast<T*>(temp_storage)));
  */
  NewKernelUtil<device_type, T>::RowMax(ctx, n, w, prob, tmp, temp_storage, temp_storage_bytes);

  // sub | every element of prob blob subract the max value of the same sample
  SoftmaxKernelUtil<device_type, T>::Sub(ctx, n, w, prob, tmp);
  // exp | exponentiation every element
  NewKernelUtil<device_type, T>::Exp(ctx, n * w, prob, prob);
  // sum | calculate sum of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the tmp data after exp
  /* ndarray reduce sum
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx, XpuVarNdarray<T>({n, 1}, tmp), XpuVarNdarray<const T>({n, w}, prob),
      XpuVarNdarray<T>({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                       reinterpret_cast<T*>(temp_storage)));
  */
  NewKernelUtil<device_type, T>::RowSum(ctx, n, w, prob, tmp, temp_storage, temp_storage_bytes);
  // div | every element of prob[i] divided by the data of tmp[i] (the sum
  // value)
  SoftmaxKernelUtil<device_type, T>::Div(ctx, n, w, prob, tmp);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
