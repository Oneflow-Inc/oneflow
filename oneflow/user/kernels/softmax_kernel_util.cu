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
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

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

constexpr int64_t kSoftmaxGpuBlockSize = 128;

int64_t GetSoftmaxBlockSize() { return kSoftmaxGpuBlockSize; }

int64_t GetSoftmaxNumBlocks(const int64_t num_instances) {
  return std::min(static_cast<int32_t>(num_instances), kCudaMaxBlocksNum);
}

template<typename T>
struct ComputeDataType {
  using type = T;
};

template<>
struct ComputeDataType<half> {
  using type = float;
};

template<typename T>
__global__ void BroadcastSubExpGpuImpl(const int64_t num_instances, const int64_t num_classes,
                                       const T* x, const T* y, T* z) {
  using ComputeType = typename ComputeDataType<T>::type;
  const int64_t tid = threadIdx.x;
  __shared__ ComputeType row_sub;
  for (int64_t row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int64_t row_offset = row * num_classes;
    const T* x_row = x + row_offset;
    T* z_row = z + row_offset;
    if (tid == 0) { row_sub = static_cast<ComputeType>(y[row]); }
    __syncthreads();
    const ComputeType row_sub_t = row_sub;
    for (int64_t col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      z_row[col] = static_cast<T>(
          ExpFunctor<ComputeType>::Forward(static_cast<ComputeType>(x_row[col]) - row_sub_t));
    }
  }
}

template<typename T>
int64_t GetMinNumClasses() {
  return 32;
}

template<typename T>
void BroadcastSubExpGpu(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                        const T* x, const T* y, T* z) {
  BroadcastSubExpGpuImpl<<<GetSoftmaxNumBlocks(num_instances), GetSoftmaxBlockSize(), 0,
                           ctx->cuda_stream()>>>(num_instances, num_classes, x, y, z);
}

template<>
void BroadcastSubExpGpu<float16>(DeviceCtx* ctx, const int64_t num_instances,
                                 const int64_t num_classes, const float16* x, const float16* y,
                                 float16* z) {
  BroadcastSubExpGpu<half>(ctx, num_instances, num_classes, reinterpret_cast<const half*>(x),
                           reinterpret_cast<const half*>(y), reinterpret_cast<half*>(z));
}

}  // namespace

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kGPU, T> {
  static size_t GetComputeProbTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetProbTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }

  static size_t GetComputeDiffTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetDiffTmpSize<T>(n, w) + GetReduceTempStorageSize<T>(n, w);
  }

  static void ComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* prob,
                          void* temp_storage, const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kGPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kGPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        SoftmaxKernelUtil<DeviceType::kGPU, T>::GetComputeProbTempStorageSizeInBytes(n, w);
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
    // sub | prob[i][j] = in[i][j] - tmp[i]
    // exp | prob[i][j] = exp(prob[i][j])
    if (w >= GetMinNumClasses<T>()) {
      BroadcastSubExpGpu(ctx, n, w, in, tmp, prob);
    } else {
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastSub(ctx, Var({n, w}, prob), Val({n, w}, in),
                                                     Val({n, 1}, tmp));
      NdarrayUtil<DeviceType::kGPU, T>::InplaceExp(ctx, Var({n, w}, prob));
    }

    // sum | tmp[i] = Sum_j(prob[i][j])
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, prob),
                                                reduce_storage_var);
    // div | prob[i][j] /= tmp[i]
    NdarrayUtil<DeviceType::kGPU, T>::InplaceBroadcastDiv(ctx, Var({n, w}, prob), Val({n, 1}, tmp));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes) {
    auto Val = NdarrayUtil<DeviceType::kGPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kGPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes =
        SoftmaxKernelUtil<DeviceType::kGPU, T>::GetComputeProbTempStorageSizeInBytes(n, w);
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
    NdarrayUtil<DeviceType::kGPU, T>::Mul(ctx, Var({n * w}, tmp), Val({n * w}, out),
                                          Val({n * w}, dy));
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, tmp),
                                                reduce_storage_var);
    // sub | dx[i][j] = dy[i][j] - sum_vec[i]
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                                   Val({n, 1}, sum_vec));
    // elementwise multiplication | dx[i][j] *= out[i][j]
    NdarrayUtil<DeviceType::kGPU, T>::InplaceMul(ctx, Var({n * w}, dx), Val({n * w}, out));
  }
};

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(data_type) \
  template struct SoftmaxKernelUtil<DeviceType::kGPU, data_type>;
INSTANTIATE_SOFTMAX_KERNEL_UTIL(float16)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(float)
INSTANTIATE_SOFTMAX_KERNEL_UTIL(double)
#undef INSTANTIATE_SOFTMAX_KERNEL_UTIL
}  // namespace oneflow
