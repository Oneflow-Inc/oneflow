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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/softmax_kernel_log_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

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



template<typename T>
struct LogSoftmaxKernelUtil<DeviceType::kCPU, T> {
  static size_t GetComputeProbTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetProbTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }

  static size_t GetComputeDiffTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetDiffTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }
   
  static void ComputeOut(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* out,
                          void* temp_storage, const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        LogSoftmaxKernelUtil<DeviceType::kCPU, T>::GetComputeProbTempStorageSizeInBytes(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
    const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
    T* reduce_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_storage_var =
        Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
    T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                  + reduce_temp_storage_bytes);
    // max | tmp[i] = Max_j(in[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceMax(ctx, Var({n, 1}, tmp), Val({n, w}, in),
                                                reduce_storage_var);
    // sub | out[i][j] = in[i][j] - tmp[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(ctx, Var({n, w}, out), Val({n, w}, in),
                                                   Val({n, 1}, tmp));
    // exp | out[i][j] = exp(out[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::InplaceExp(ctx,Var({n,w},out));
    // sum | tmp[i] = Sum_j(out[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, out),
                                                reduce_storage_var);
    // log | out[i][j] = log(out[i][j])
    FOR_RANGE(int64_t, i, 0, n) {
        FOR_RANGE(int64_t, j, 0, w) {
             out[i * w + j] = SafeLog(out[i * w + j]);
        }
    }
    // tmp | tmp[i] = log(tmp[i])
    FOR_RANGE(int64_t, i, 0, n){
       tmp[i] = SafeLog(tmp[i]);
    }
    // sub | out[i][j] -= tmp[i]
    NdarrayUtil<DeviceType::kCPU, T>::InplaceBroadcastSub(ctx, Var({n, w}, out), Val({n, 1}, tmp));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        SoftmaxKernelUtil<DeviceType::kCPU, T>::GetComputeProbTempStorageSizeInBytes(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
    const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
    T* reduce_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_storage_var =
        Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
    T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                  + reduce_temp_storage_bytes);
    
    //sum | dx[i] = Sum_j(dy[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, dx), Val({n, w}, dy),
                                                reduce_storage_var);
    //mul | tmp[i][j] = out[i][j]*dx[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastMul(ctx, Var({n, w}, tmp), Val({n, w}, out),
                                                Val({n, 1}, dx));

    //sum | dx[i][j] = out[i][j]-tmp[i][j]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                                   Val({n, w}, tmp));
  }
};

#define INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(data_type) \
  template struct LogSoftmaxKernelUtil<DeviceType::kCPU, data_type>;
INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(float)
INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL(double)
#undef INSTANTIATE_LOGSOFTMAX_KERNEL_UTIL
}  // namespace oneflow
