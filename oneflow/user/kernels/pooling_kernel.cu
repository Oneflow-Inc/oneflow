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

template<typename T>
__global__ void DoCUDAMaxPool2dForward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                       int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                       int32_t padding_h, int32_t padding_w, int64_t n_batch,
                                       int64_t n_channel, int64_t x_height, int64_t x_width,
                                       int64_t y_height, int64_t y_width, int32_t kernel_size_h,
                                       int32_t kernel_size_w, int32_t stride_h, int32_t stride_w,
                                       int32_t dilation_h, int32_t dilation_w) {
  Maxpool2dFarwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, padding_h, padding_w,
                             n_batch, n_channel, x_height, x_width, y_height, y_width,
                             kernel_size_h, kernel_size_w, stride_h, stride_w, dilation_h,
                             dilation_w);
};

template<typename T>
__global__ void DoCUDAMaxPool3dForward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                       int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                       int32_t padding_t, int32_t padding_h, int32_t padding_w,
                                       int64_t n_batch, int64_t n_channel, int64_t x_time,
                                       int64_t x_height, int64_t x_width, int64_t y_time,
                                       int64_t y_height, int64_t y_width, int32_t kernel_size_t,
                                       int32_t kernel_size_h, int32_t kernel_size_w,
                                       int32_t stride_t, int32_t stride_h, int32_t stride_w,
                                       int32_t dilation_t, int32_t dilation_h, int32_t dilation_w) {
  Maxpool3dFarwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, padding_t, padding_h,
                             padding_w, n_batch, n_channel, x_time, x_height, x_width, y_time,
                             y_height, y_width, kernel_size_t, kernel_size_h, kernel_size_w,
                             stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w);
};

template<typename T>
__global__ void DoCUDAMaxPool2dBackward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                        const int64_t elem_num, const T* src, T* dest,
                                        const int64_t* indice_ptr, const int64_t n_batch,
                                        const int64_t n_channel, const int64_t src_height,
                                        const int64_t src_width, const int64_t dst_height,
                                        const int64_t dst_width) {
  Maxpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_height, src_width, dst_height, dst_width);
};

template<typename T>
__global__ void DoCUDAMaxPool3dBackward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                        const int64_t elem_num, const T* src, T* dest,
                                        const int64_t* indice_ptr, const int64_t n_batch,
                                        const int64_t n_channel, const int64_t src_time,
                                        const int64_t src_height, const int64_t src_width,
                                        const int64_t dst_time, const int64_t dst_height,
                                        const int64_t dst_width) {
  Maxpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_time, src_height, src_width, dst_time, dst_height, dst_width);
};

template<typename T>
struct PoolingKernelUtil<DeviceType::kGPU, T> {
  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t& elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    DoCUDAMaxPool2dForward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.padding_before_3d()[1],
            params_3d.padding_before_3d()[2], params_3d.num_batch(), params_3d.num_channel(),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
            params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2],
            params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.dilation_3d()[1],
            params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    DoCUDAMaxPool2dBackward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    DoCUDAMaxPool3dForward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.padding_before_3d()[0],
            params_3d.padding_before_3d()[1], params_3d.padding_before_3d()[2],
            params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(2),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
            params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
            params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[0],
            params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2],
            params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
            params_3d.dilation_3d()[0], params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    DoCUDAMaxPool3dBackward<T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
            params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(2),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kGPU),
                                 POOLING_DATA_TYPE_GPU_SEQ);

}  // namespace oneflow
#endif  // WITH_CUDA
