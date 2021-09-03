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
#include "oneflow/user/kernels/logsoftmax_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ComputeLogGpu(const int64_t len, T* out, T* in) {
  CUDA_1D_KERNEL_LOOP(i, len) { out[i] = SafeLog(in[i]); }
}

}  // namespace

template<typename T>
struct LogSoftmaxKernelUtil<DeviceType::kGPU, T> {
  static size_t GetComputeProbTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetProbTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }

  static size_t GetComputeDiffTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetDiffTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }

  static void ComputeOut(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* prob,
                         T* out, void* temp_storage, const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kGPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kGPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        LogSoftmaxKernelUtil<DeviceType::kGPU, T>::GetComputeProbTempStorageSizeInBytes(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
    const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
    T* reduce_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_storage_var =
        Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
    T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                  + reduce_temp_storage_bytes);
    // max | tmp[i] = Max_j(in[i][j])
    NdarrayUtil<DeviceType::kGPU, T>::ReduceMax(ctx, Var({n, 1}, tmp), Val({n, w}, in),
                                                reduce_storage_var);
    // sub | out[i][j] = in[i][j] - tmp[i]
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastSub(ctx, Var({n, w}, out), Val({n, w}, in),
                                                   Val({n, 1}, tmp));
    // exp | prob[i][j] = exp(out[i][j])
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastExp(ctx, Var({n, w}, prob), Val({n, w}, out));
    // sum | tmp[i] = Sum_j(prob[i][j])
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, prob),
                                                reduce_storage_var);
    // div | prob[i][j] /= tmp[i]
    NdarrayUtil<DeviceType::kGPU, T>::InplaceBroadcastDiv(ctx, Var({n, w}, prob), Val({n, 1}, tmp));
    // tmp | tmp[i] = log(tmp[i])
    ComputeLogGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, tmp, tmp);
    // sub | out[i][j] -= tmp[i]
    NdarrayUtil<DeviceType::kGPU, T>::InplaceBroadcastSub(ctx, Var({n, w}, out), Val({n, 1}, tmp));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kGPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kGPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        LogSoftmaxKernelUtil<DeviceType::kGPU, T>::GetComputeDiffTempStorageSizeInBytes(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
    const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
    T* reduce_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_storage_var =
        Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
    T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                  + reduce_temp_storage_bytes);
    // sum | dx[i] = Sum_j(dy[i][j])
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(ctx, Var({n, 1}, dx), Val({n, w}, dy),
                                                reduce_storage_var);
    // mul | tmp[i][j] = out[i][j]*dx[i]
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(ctx, Var({n, w}, tmp), Val({n, w}, out),
                                                   Val({n, 1}, dx));
    // sum | dx[i][j] = out[i][j]-tmp[i][j]
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                                   Val({n, w}, tmp));
  }
};

#define INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(data_type) \
  template struct LogSoftmaxKernelUtil<DeviceType::kGPU, data_type>;
INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(half)
INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(float)
INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(double)
#undef INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL
}  // namespace oneflow
