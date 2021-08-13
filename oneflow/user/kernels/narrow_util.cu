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
#include "oneflow/user/kernels/narrow_util.h"

namespace oneflow {

namespace {

template<typename T, typename IDX>
__global__ void NarrowForwardGpu(const IDX elem_cnt, const int64_t start, const int64_t length,
                                 const T* in, const IDX narrow_dim_size, const IDX inner_dim_size,
                                 T* out) {
  const IDX outer_dim_elem_cnt = length * inner_dim_size;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    const IDX outer_idx = i / outer_dim_elem_cnt;
    const IDX inner_idx = i % inner_dim_size;
    const IDX indices_idx = i % outer_dim_elem_cnt / inner_dim_size;
    const IDX offset = outer_idx * narrow_dim_size * inner_dim_size
                       + (start + indices_idx) * inner_dim_size + inner_idx;
    out[i] = in[offset];
  }
}

template<typename T, typename IDX>
__global__ void NarrowBackwardGpu(const IDX elem_cnt, const int64_t start, const int64_t length,
                                  const T* dy, const IDX narrow_dim_size, const IDX inner_dim_size,
                                  T* dx) {
  const IDX outer_dim_elem_cnt = length * inner_dim_size;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    const IDX outer_idx = i / outer_dim_elem_cnt;
    const IDX inner_idx = i % inner_dim_size;
    const IDX indices_idx = i % outer_dim_elem_cnt / inner_dim_size;
    const IDX offset = outer_idx * narrow_dim_size * inner_dim_size
                       + (start + indices_idx) * inner_dim_size + inner_idx;
    dx[offset] = dy[i];
  }
}

bool IsSafeUseIndex32(const Shape& flat_in_shape, const int64_t& length) {
  const int64_t in_elem_cnt = flat_in_shape.elem_cnt();
  const int64_t out_elem_cnt = flat_in_shape.At(0) * length * flat_in_shape.At(2);
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

}  // namespace

template<typename T>
struct NarrowKernelUtil<DeviceType::kGPU, T> final {
  static void Forward(DeviceCtx* ctx, const T* in, const Shape& flat_in_shape, T* out,
                      const int64_t& start, const int64_t& length) {
    const int64_t out_elem_cnt = flat_in_shape.At(0) * length * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, length)) {
      NarrowForwardGpu<T, int32_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, start, length, in, flat_in_shape.At(1), flat_in_shape.At(2), out);
    } else {
      NarrowForwardGpu<T, int64_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, start, length, in, flat_in_shape.At(1), flat_in_shape.At(2), out);
    }
  }

  static void Backward(DeviceCtx* ctx, const T* dy, const Shape& flat_in_shape, T* dx,
                       const int64_t& start, const int64_t& length) {
    const int64_t out_elem_cnt = flat_in_shape.At(0) * length * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, length)) {
      NarrowBackwardGpu<T, int32_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, start, length, dy, flat_in_shape.At(1), flat_in_shape.At(2), dx);
    } else {
      NarrowBackwardGpu<T, int64_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, start, length, dy, flat_in_shape.At(1), flat_in_shape.At(2), dx);
    }
  }
};

INSTANTIATE_NARROW_KERNEL_UTIL_WITH_DEVICE(DeviceType::kGPU)

}  // namespace oneflow
