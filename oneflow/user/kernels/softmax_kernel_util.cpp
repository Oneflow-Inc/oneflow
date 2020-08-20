/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/kernels/softmax_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<typename T>
size_t GetProbTmpSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * sizeof(T));
}

template<typename T>
size_t GetDiffTmpSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * sizeof(T));
}

template<typename T>
size_t GetReduceTempStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * w * sizeof(T));
}

}  // namespace

template<DeviceType device_type, typename T>
size_t SoftmaxKernelUtil<device_type, T>::GetComputeProbTempStorageSizeInBytes(int64_t n,
                                                                               int64_t w) {
  return GetProbTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
}

template<DeviceType device_type, typename T>
size_t SoftmaxKernelUtil<device_type, T>::GetComputeDiffTempStorageSizeInBytes(int64_t n,
                                                                               int64_t w) {
  return GetDiffTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
}

template<DeviceType device_type, typename T>
void SoftmaxKernelUtil<device_type, T>::ComputeProb(DeviceCtx* ctx, const int64_t n,
                                                    const int64_t w, const T* in, T* prob,
                                                    void* temp_storage,
                                                    const size_t temp_storage_bytes) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();
  const size_t min_temp_storage_bytes =
      SoftmaxKernelUtil<device_type, T>::GetComputeProbTempStorageSizeInBytes(n, w);
  CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
  const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
  T* reduce_storage = reinterpret_cast<T*>(temp_storage);
  auto reduce_storage_var =
      Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
  T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                + reduce_temp_storage_bytes);
  // max | tmp[i] = Max_j(in[i][j])
  NdarrayUtil<device_type, T>::ReduceMax(ctx, Var({n, 1}, tmp), Val({n, w}, in),
                                         reduce_storage_var);
  // sub | prob[i][j] = in[i][j] - tmp[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({n, w}, prob), Val({n, w}, in),
                                            Val({n, 1}, tmp));
  // exp | prob[i][j] = exp(prob[i][j])
  NdarrayUtil<device_type, T>::InplaceExp(ctx, Var({n, w}, prob));
  // sum | tmp[i] = Sum_j(prob[i][j])
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, prob),
                                         reduce_storage_var);
  // div | prob[i][j] /= tmp[i]
  NdarrayUtil<device_type, T>::InplaceBroadcastDiv(ctx, Var({n, w}, prob), Val({n, 1}, tmp));
}

template<DeviceType device_type, typename T>
void SoftmaxKernelUtil<device_type, T>::ComputeDiff(DeviceCtx* ctx, const int64_t n,
                                                    const int64_t w, const T* dy, const T* out,
                                                    T* dx, void* temp_storage,
                                                    const size_t temp_storage_bytes) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();
  const size_t min_temp_storage_bytes =
      SoftmaxKernelUtil<device_type, T>::GetComputeProbTempStorageSizeInBytes(n, w);
  CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
  const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
  T* reduce_storage = reinterpret_cast<T*>(temp_storage);
  auto reduce_storage_var =
      Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
  T* sum_vec = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                    + reduce_temp_storage_bytes);
  // it's safe to use dx as tmp
  // dot product | get dot product sum_vec[i] from out[i] * dy[i]
  T* tmp = dx;
  NdarrayUtil<device_type, T>::Mul(ctx, Var({n * w}, tmp), Val({n * w}, out), Val({n * w}, dy));
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, tmp),
                                         reduce_storage_var);
  // sub | dx[i][j] = dy[i][j] - sum_vec[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                            Val({n, 1}, sum_vec));
  // elementwise multiplication | dx[i][j] *= out[i][j]
  NdarrayUtil<device_type, T>::InplaceMul(ctx, Var({n * w}, dx), Val({n * w}, out));
}

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(device_type, data_type) \
  template struct SoftmaxKernelUtil<device_type, data_type>;
#ifdef WITH_CUDA
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, float16)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, float)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kGPU, double)
#endif
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kCPU, float)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(DeviceType::kCPU, double)
#undef INSTANTIATE_SOFTMAX_KERNEL_UTIL
}  // namespace oneflow
