#include "oneflow/customized/kernels/softmax_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SoftmaxKernelUtil<device_type, T>::ComputeProb(DeviceCtx* ctx, const int64_t n,
                                                    const int64_t w, const T* in, T* tmp, T* prob,
                                                    void* temp_storage,
                                                    const size_t temp_storage_bytes) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();
  // max | tmp[i] = Max_j(in[i][j])
  NdarrayUtil<device_type, T>::ReduceMax(ctx, Var({n, 1}, tmp), Val({n, w}, in),
                                         Var({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                                             reinterpret_cast<T*>(temp_storage)));
  // sub | prob[i][j] = in[i][j] - tmp[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({n, w}, prob), Val({n, w}, in),
                                            Val({n, 1}, tmp));
  // exp | prob[i][j] = exp(prob[i][j])
  NdarrayUtil<device_type, T>::InplaceExp(ctx, Var({n, w}, prob));
  // sum | tmp[i] = Sum_j(prob[i][j])
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, prob),
                                         Var({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                                             reinterpret_cast<T*>(temp_storage)));
  // div | prob[i][j] /= tmp[i]
  NdarrayUtil<device_type, T>::InplaceBroadcastDiv(ctx, Var({n, w}, prob), Val({n, 1}, tmp));
}

template<DeviceType device_type, typename T>
void SoftmaxKernelUtil<device_type, T>::ComputeDiff(DeviceCtx* ctx, const int64_t n,
                                                    const int64_t w, const T* dy, const T* out,
                                                    T* sum_vec, T* dx, void* temp_storage,
                                                    const size_t temp_storage_bytes) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();
  // it's safe to use dx as tmp
  // dot product | get dot product sum_vec[i] from out[i] * dy[i]
  T* tmp = dx;
  NdarrayUtil<device_type, T>::Mul(ctx, Var({n * w}, tmp), Val({n * w}, out), Val({n * w}, dy));
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, tmp),
                                         Var({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                                             reinterpret_cast<T*>(temp_storage)));
  // sub | dx[i][j] = dy[i][j] - sum_vec[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                            Val({n, 1}, sum_vec));
  // elementwise multiplication | dx[i][j] *= out[i][j]
  NdarrayUtil<device_type, T>::InplaceMul(ctx, Var({n * w}, dx), Val({n * w}, out));
}

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(device_type, data_type) \
  template struct SoftmaxKernelUtil<device_type, data_type>;
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, float16)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, float)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, double)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kCPU, float)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kCPU, double)
#undef INSTANTIATE_SOFTMAX_KERNEL_UTIL
}  // namespace oneflow
