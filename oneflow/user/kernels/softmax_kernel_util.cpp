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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/softmax_kernel_util.h"

namespace oneflow {

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kCPU, SoftmaxAlgorithm::kSoftmax, T> {
  static void ComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* prob,
                          void* temp_storage, const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();

    const size_t min_temp_storage_bytes =
        SoftmaxComputeProbTempStorageSize<T, SoftmaxAlgorithm::kSoftmax>(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);

    const size_t reduce_operation_storage_bytes = SoftmaxReduceOperationStorageSize<T>(n, w);
    T* reduce_operation_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_operation_storage_var =
        Var({static_cast<int64_t>(reduce_operation_storage_bytes / sizeof(T))},
            reduce_operation_storage);
    T* reduce_result_storage = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                                    + reduce_operation_storage_bytes);
    // max | reduce_result_storage[i] = Max_j(in[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceMax(ctx, Var({n, 1}, reduce_result_storage),
                                                Val({n, w}, in), reduce_operation_storage_var);
    // sub | prob[i][j] = in[i][j] - reduce_result_storage[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(ctx, Var({n, w}, prob), Val({n, w}, in),
                                                   Val({n, 1}, reduce_result_storage));
    // exp | prob[i][j] = exp(prob[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::InplaceExp(ctx, Var({n, w}, prob));
    // sum | reduce_result_storage[i] = Sum_j(prob[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, reduce_result_storage),
                                                Val({n, w}, prob), reduce_operation_storage_var);
    // div | prob[i][j] /= reduce_result_storage[i]
    NdarrayUtil<DeviceType::kCPU, T>::InplaceBroadcastDiv(ctx, Var({n, w}, prob),
                                                          Val({n, 1}, reduce_result_storage));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();

    const size_t min_temp_storage_bytes =
        SoftmaxComputeDiffTempStorageSize<T, SoftmaxAlgorithm::kSoftmax>(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);

    const size_t reduce_operation_storage_bytes = SoftmaxReduceOperationStorageSize<T>(n, w);
    T* reduce_operation_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_operation_storage_var =
        Var({static_cast<int64_t>(reduce_operation_storage_bytes / sizeof(T))},
            reduce_operation_storage);
    T* sum_vec = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                      + reduce_operation_storage_bytes);
    // it's safe to use dx as tmp
    // dot product | get dot product sum_vec[i] from out[i] * dy[i]
    T* tmp = dx;
    NdarrayUtil<DeviceType::kCPU, T>::Mul(ctx, Var({n * w}, tmp), Val({n * w}, out),
                                          Val({n * w}, dy));
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, tmp),
                                                reduce_operation_storage_var);
    // sub | dx[i][j] = dy[i][j] - sum_vec[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                                   Val({n, 1}, sum_vec));
    // elementwise multiplication | dx[i][j] *= out[i][j]
    NdarrayUtil<DeviceType::kCPU, T>::InplaceMul(ctx, Var({n * w}, dx), Val({n * w}, out));
  }
};

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kCPU, SoftmaxAlgorithm::kLogSoftmax, T> {
  static void ComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* prob,
                          void* temp_storage, const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();

    const size_t min_temp_storage_bytes =
        SoftmaxComputeProbTempStorageSize<T, SoftmaxAlgorithm::kLogSoftmax>(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);

    const size_t reduce_operation_storage_bytes = SoftmaxReduceOperationStorageSize<T>(n, w);
    const size_t reduce_result_storage_bytes = SoftmaxReduceResultStorageSize<T>(n);

    T* reduce_operation_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_operation_storage_var =
        Var({static_cast<int64_t>(reduce_operation_storage_bytes / sizeof(T))},
            reduce_operation_storage);
    T* reduce_result_storage = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                                    + reduce_operation_storage_bytes);
    T* new_temp_storage =
        reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                             + reduce_operation_storage_bytes + reduce_result_storage_bytes);

    // max | reduce_result_storage[i] = Max_j(in[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceMax(ctx, Var({n, 1}, reduce_result_storage),
                                                Val({n, w}, in), reduce_operation_storage_var);
    // sub | new_temp_storage[i][j] = in[i][j] - reduce_result_storage[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(
        ctx, Var({n, w}, new_temp_storage), Val({n, w}, in), Val({n, 1}, reduce_result_storage));
    // exp | prob[i][j] = Exp(new_temp_storage[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastExp(ctx, Var({n, w}, prob),
                                                   Val({n, w}, new_temp_storage));
    // sum / reduce_result_storage[i] = Sum_j(prob[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, reduce_result_storage),
                                                Val({n, w}, prob), reduce_operation_storage_var);
    // log | reduce_result_storage[i] = SafeLog(reduce_result_storage[i])
    FOR_RANGE(int64_t, i, 0, n) { reduce_result_storage[i] = SafeLog(reduce_result_storage[i]); }
    // sub / prob[i][j] = new_temp_storage[i][j] - reduce_result_storage[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(
        ctx, Var({n, w}, prob), Val({n, w}, new_temp_storage), Val({n, 1}, reduce_result_storage));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();

    const size_t min_temp_storage_bytes =
        SoftmaxComputeDiffTempStorageSize<T, SoftmaxAlgorithm::kLogSoftmax>(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);

    const size_t reduce_operation_storage_bytes = SoftmaxReduceOperationStorageSize<T>(n, w);
    const size_t reduce_result_storage_bytes = SoftmaxReduceResultStorageSize<T>(n);

    T* reduce_operation_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_operation_storage_var =
        Var({static_cast<int64_t>(reduce_operation_storage_bytes / sizeof(T))},
            reduce_operation_storage);
    T* sum_vec = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                      + reduce_operation_storage_bytes);
    T* new_temp_storage =
        reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                             + reduce_operation_storage_bytes + reduce_result_storage_bytes);
    // sum / sum_vec[i] = Sum_i(dy[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, dy),
                                                reduce_operation_storage_var);
    // exp / new_temp_storage[i][j] = Exp(out[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastExp(ctx, Var({n, w}, new_temp_storage),
                                                   Val({n, w}, out));
    // mul / new_temp_storage[i][j] = new_temp_storage[i][j] * sum_vec[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastMul(
        ctx, Var({n, w}, new_temp_storage), Val({n, w}, new_temp_storage), Val({n, 1}, sum_vec));
    // sub / dx[i][j] = dy[i][j] - new_temp_storage[i][j]
    NdarrayUtil<DeviceType::kCPU, T>::Sub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                          Val({n, w}, new_temp_storage));
  }
};

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(data_type, softmax_algorithm) \
  template struct SoftmaxKernelUtil<DeviceType::kCPU, softmax_algorithm, data_type>;
INSTANTIATE_SOFTMAX_KERNEL_UTIL(float, SoftmaxAlgorithm::kSoftmax)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(double, SoftmaxAlgorithm::kSoftmax)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(float, SoftmaxAlgorithm::kLogSoftmax)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(double, SoftmaxAlgorithm::kLogSoftmax)
#undef INSTANTIATE_SOFTMAX_KERNEL_UTIL
}  // namespace oneflow
