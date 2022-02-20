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
#ifndef ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_CUH_
#define ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_CUH_

#if defined(__CUDACC__)

#include "oneflow/user/kernels/multi_reduce_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int64_t kMultiReduceMaxPackSize = 64;

template<typename T>
struct MultiReduceParamsPack {
  MultiReduceParam<T> params[kMultiReduceMaxPackSize];
  size_t size;
};

template<typename T, typename F>
__global__ void MultiReduceSumGpu(F func, const MultiReduceParamsPack<T> pack_params, T* sum) {
  T t_sum = 0;
  for (int i = 0; i < pack_params.size; ++i) {
    const auto& param = pack_params.params[i];
    CUDA_1D_KERNEL_LOOP(j, param.size) { t_sum += func(param.data[j]); }
  }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T b_sum = BlockReduce(temp_storage).Sum(t_sum);
  if (threadIdx.x == 0) { cuda::atomic::Add(sum, b_sum); }
}

}  // namespace

template<typename T, typename F>
struct MultiReduceSum<DeviceType::kCUDA, T, F> {
  void operator()(ep::Stream* stream, F func, const std::vector<MultiReduceParam<T>>& params,
                  T* sum) {
    Memset<DeviceType::kCUDA>(stream, sum, 0, sizeof(T));
    for (size_t i = 0; i < params.size(); i += kMultiReduceMaxPackSize) {
      MultiReduceParamsPack<T> pack_params{};
      size_t max_elem_cnt = 0;
      pack_params.size = std::min(kMultiReduceMaxPackSize, params.size() - i);
      for (size_t j = 0; j < pack_params.size; ++j) {
        pack_params.params[j] = params[i + j];
        max_elem_cnt = std::max(max_elem_cnt, pack_params.params[j].size);
      }
      MultiReduceSumGpu<T, F>
          <<<BlocksNum4ThreadsNum(max_elem_cnt), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(func, pack_params, sum);
    }
  }
}

template<>
struct Abs<half> {
  OF_DEVICE_FUNC half operator()(half x) const {
    return __hlt(x, GetZeroVal<half>()) ? __hneg(x) : x;
  }
};

template<>
struct PowByZero {
  OF_DEVICE_FUNC half operator()(half x) const {
    return x == GetZeroVal<half>() ? x : GetOneVal<half>();
  }
};

template<>
struct AbsPow<half> {
  explicit AbsPow(float base) : base_(base) {}

  OF_DEVICE_FUNC half operator()(half x) {
    half abs_x = __hlt(x, GetZeroVal<half>()) ? __hneg(x) : x;
    return __float2half(pow(__half2float(x), base_));
  }

 private:
  float base_;
};

}  // namespace oneflow

#endif  // defined(__CUDACC__)

#endif  // ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_CUH_
