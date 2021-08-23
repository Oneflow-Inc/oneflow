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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/pooling_kernel_util.h"

namespace oneflow {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool1dForward(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                int32_t padding_l, int64_t n_batch, int64_t n_channel,
                                int64_t x_length, int64_t y_length, int32_t kernel_size_l,
                                int32_t stride_l, int32_t dilation_l) {
  Maxpool1dForwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, padding_l, n_batch,
                             n_channel, x_length, y_length, kernel_size_l, stride_l, dilation_l);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool2dForward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                int32_t padding_h, int32_t padding_w, int64_t n_batch,
                                int64_t n_channel, int64_t x_height, int64_t x_width,
                                int64_t y_height, int64_t y_width, int32_t kernel_size_h,
                                int32_t kernel_size_w, int32_t stride_h, int32_t stride_w,
                                int32_t dilation_h, int32_t dilation_w) {
  Maxpool2dForwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, padding_h, padding_w,
                             n_batch, n_channel, x_height, x_width, y_height, y_width,
                             kernel_size_h, kernel_size_w, stride_h, stride_w, dilation_h,
                             dilation_w);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool3dForward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                int32_t padding_t, int32_t padding_h, int32_t padding_w,
                                int64_t n_batch, int64_t n_channel, int64_t x_time,
                                int64_t x_height, int64_t x_width, int64_t y_time, int64_t y_height,
                                int64_t y_width, int32_t kernel_size_t, int32_t kernel_size_h,
                                int32_t kernel_size_w, int32_t stride_t, int32_t stride_h,
                                int32_t stride_w, int32_t dilation_t, int32_t dilation_h,
                                int32_t dilation_w) {
  Maxpool3dForwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, padding_t, padding_h,
                             padding_w, n_batch, n_channel, x_time, x_height, x_width, y_time,
                             y_height, y_width, kernel_size_t, kernel_size_h, kernel_size_w,
                             stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool1dBackward(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                 const int64_t elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t n_batch,
                                 const int64_t n_channel, const int64_t src_length,
                                 const int64_t dst_length) {
  Maxpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_length, dst_length);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool2dBackward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                 const int64_t elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t n_batch,
                                 const int64_t n_channel, const int64_t src_height,
                                 const int64_t src_width, const int64_t dst_height,
                                 const int64_t dst_width) {
  Maxpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                              src_height, src_width, dst_height, dst_width);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool3dBackward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
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
  static void Maxpool1dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    DoCUDAMaxPool1dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[2],
            params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(4),
            params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2],
            params_3d.dilation_3d()[2]);
  }

  static void Maxpool1dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    DoCUDAMaxPool1dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    DoCUDAMaxPool2dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
            params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
            params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2],
            params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.dilation_3d()[1],
            params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    DoCUDAMaxPool2dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    DoCUDAMaxPool3dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[0],
            params_3d.padding()[1], params_3d.padding()[2], params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
            params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(2),
            params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[0], params_3d.pooling_size_3d()[1],
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
            params_3d.stride_3d()[2], params_3d.dilation_3d()[0], params_3d.dilation_3d()[1],
            params_3d.dilation_3d()[2]);
  }

  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    DoCUDAMaxPool3dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
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
