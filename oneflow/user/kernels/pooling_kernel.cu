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
#include <cstdint>
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/user/kernels/pooling_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

// Fill ShapeView into dim vector
DimVector ShapeViewToDimVector(const ShapeView& tensor_shape) {
  int64_t ndims = tensor_shape.NumAxes();
  DimVector shape_vec(ndims);
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1];
  return shape_vec;
}
}  // namespace

template<typename T>
__global__ void DoCUDAMaxPool2dForward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                       int64_t elem_num, T maxval, const T* src, T* dest,
                                       int64_t* indice_ptr, int32_t padding_h, int32_t padding_w,
                                       int64_t n_batch, int64_t n_channel, int64_t x_height,
                                       int64_t x_width, int64_t y_height, int64_t y_width,
                                       int32_t kernel_size_h, int32_t kernel_size_w,
                                       int32_t stride_h, int32_t stride_w, int32_t dilation_h,
                                       int32_t dilation_w, bool return_indices, bool ceil_mode) {
  Maxpool2dFarwardCompute<T>(index_helper, elem_num, maxval, src, dest, indice_ptr, padding_h,
                             padding_w, n_batch, n_channel, x_height, x_width, y_height, y_width,
                             kernel_size_h, kernel_size_w, stride_h, stride_w, dilation_h,
                             dilation_w, return_indices, ceil_mode);
};

template<typename T>
__global__ void DoCUDAMaxPool3dForward(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, T maxval, const T* src,
    T* dest, int64_t* indice_ptr, int32_t padding_t, int32_t padding_h, int32_t padding_w,
    int64_t n_batch, int64_t n_channel, int64_t x_time, int64_t x_height, int64_t x_width,
    int64_t y_time, int64_t y_height, int64_t y_width, int32_t kernel_size_t, int32_t kernel_size_h,
    int32_t kernel_size_w, int32_t stride_t, int32_t stride_h, int32_t stride_w, int32_t dilation_t,
    int32_t dilation_h, int32_t dilation_w, bool return_indices, bool ceil_mode) {
  Maxpool3dFarwardCompute<T>(index_helper, elem_num, maxval, src, dest, indice_ptr, padding_t,
                             padding_h, padding_w, n_batch, n_channel, x_time, x_height, x_width,
                             y_time, y_height, y_width, kernel_size_t, kernel_size_h, kernel_size_w,
                             stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
                             return_indices, ceil_mode);
};

template<typename T>
__global__ void DoCUDAMaxPool2dBackward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                        const int64_t elem_num, const T* src, T* dest,
                                        const int64_t* indice_ptr, const int64_t n_batch,
                                        const int64_t n_channel, const int64_t src_height,
                                        const int64_t src_width, const int64_t dst_height,
                                        const int64_t dst_width, const bool return_indices,
                                        const bool ceil_mode) {
  Maxpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_height, src_width, dst_height, dst_width, return_indices,
                              ceil_mode);
};

template<typename T>
__global__ void DoCUDAMaxPool3dBackward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                        const int64_t elem_num, const T* src, T* dest,
                                        const int64_t* indice_ptr, const int64_t n_batch,
                                        const int64_t n_channel, const int64_t src_time,
                                        const int64_t src_height, const int64_t src_width,
                                        const int64_t dst_time, const int64_t dst_height,
                                        const int64_t dst_width, const bool return_indices,
                                        const bool ceil_mode) {
  Maxpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_time, src_height, src_width, dst_time, dst_height, dst_width,
                              return_indices, ceil_mode);
};

template<typename T>
struct PoolingKernelUtil<DeviceType::kGPU, T> {
  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const std::vector<int32_t> padding_before, const int64_t n_batch,
                               const int64_t n_channel, const int64_t x_height,
                               const int64_t x_width, const int64_t y_height, const int64_t y_width,
                               const std::vector<int32_t> kernel_size,
                               const std::vector<int32_t> stride,
                               const std::vector<int32_t> dilation, const bool return_indices,
                               const bool ceil_mode) {
    T maxval = -std::numeric_limits<T>::infinity();
    DoCUDAMaxPool2dForward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, maxval, src, dest, indice_ptr, padding_before[0],
            padding_before[1], n_batch, n_channel, x_height, x_width, y_height, y_width,
            kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1],
            return_indices, ceil_mode);
  }

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch,
                                const int64_t n_channel, const int64_t src_height,
                                const int64_t src_width, const int64_t dst_height,
                                const int64_t dst_width, const bool return_indices,
                                const bool ceil_mode) {
    DoCUDAMaxPool2dBackward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel, src_height,
            src_width, dst_height, dst_width, return_indices, ceil_mode);
  }

  static void Maxpool3dForward(
      DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper, const int64_t elem_num,
      const T* src, T* dest, int64_t* indice_ptr, const std::vector<int32_t> padding_before,
      const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
      const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
      const std::vector<int32_t> kernel_size, const std::vector<int32_t> stride,
      const std::vector<int32_t> dilation, const bool return_indices, const bool ceil_mode) {
    T maxval = -std::numeric_limits<T>::infinity();
    DoCUDAMaxPool3dForward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, maxval, src, dest, indice_ptr, padding_before[0],
            padding_before[1], padding_before[2], n_batch, n_channel, x_time, x_height, x_width,
            y_time, y_height, y_width, kernel_size[0], kernel_size[1], kernel_size[2], stride[0],
            stride[1], stride[2], dilation[0], dilation[1], dilation[2], return_indices, ceil_mode);
  }

  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch,
                                const int64_t n_channel, const int64_t src_time,
                                const int64_t src_height, const int64_t src_width,
                                const int64_t dst_time, const int64_t dst_height,
                                const int64_t dst_width, const bool return_indices,
                                const bool ceil_mode) {
    DoCUDAMaxPool3dBackward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel, src_time, src_height,
            src_width, dst_time, dst_height, dst_width, return_indices, ceil_mode);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kGPU),
                                 POOLING_DATA_TYPE_GPU_SEQ);

}  // namespace oneflow
#endif  // WITH_CUDA
