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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/avg_pooling_kernel_util.h"

namespace oneflow {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(const int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

}  // namespace

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool1dForward(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                int64_t elem_num, const T* src, T* dest, int32_t padding_l,
                                int64_t n_batch, int64_t n_channel, int64_t x_length,
                                int64_t y_length, int32_t kernel_size_l, int32_t stride_l,
                                const bool count_include_pad, int64_t divisor_override) {
  Avgpool1dForwardCompute<T>(index_helper, elem_num, src, dest, padding_l, n_batch, n_channel,
                             x_length, y_length, kernel_size_l, stride_l, count_include_pad,
                             divisor_override);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool2dForward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                int64_t elem_num, const T* src, T* dest, int32_t padding_h,
                                int32_t padding_w, int64_t n_batch, int64_t n_channel,
                                int64_t x_height, int64_t x_width, int64_t y_height,
                                int64_t y_width, int32_t kernel_size_h, int32_t kernel_size_w,
                                int32_t stride_h, int32_t stride_w, const bool count_include_pad,
                                int64_t divisor_override) {
  Avgpool2dForwardCompute<T>(index_helper, elem_num, src, dest, padding_h, padding_w, n_batch,
                             n_channel, x_height, x_width, y_height, y_width, kernel_size_h,
                             kernel_size_w, stride_h, stride_w, count_include_pad,
                             divisor_override);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool3dForward(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                int64_t elem_num, const T* src, T* dest, int32_t padding_t,
                                int32_t padding_h, int32_t padding_w, int64_t n_batch,
                                int64_t n_channel, int64_t x_time, int64_t x_height,
                                int64_t x_width, int64_t y_time, int64_t y_height, int64_t y_width,
                                int32_t kernel_size_t, int32_t kernel_size_h, int32_t kernel_size_w,
                                int32_t stride_t, int32_t stride_h, int32_t stride_w,
                                const bool count_include_pad, int64_t divisor_override) {
  Avgpool3dForwardCompute<T>(index_helper, elem_num, src, dest, padding_t, padding_h, padding_w,
                             n_batch, n_channel, x_time, x_height, x_width, y_time, y_height,
                             y_width, kernel_size_t, kernel_size_h, kernel_size_w, stride_t,
                             stride_h, stride_w, count_include_pad, divisor_override);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool1dBackward(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                 int64_t elem_num, const T* src, T* dest, const int32_t padding_l,
                                 const int64_t n_batch, const int64_t n_channel,
                                 const int64_t input_length, const int64_t output_length,
                                 const int32_t kernel_size_l, const int32_t stride_l,
                                 const bool count_include_pad, int64_t divisor_override) {
  Avgpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_l, n_batch, n_channel,
                              input_length, output_length, kernel_size_l, stride_l,
                              count_include_pad, divisor_override);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool2dBackward(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                 int64_t elem_num, const T* src, T* dest, const int32_t padding_h,
                                 const int32_t padding_w, const int64_t n_batch,
                                 const int64_t n_channel, const int64_t input_height,
                                 const int64_t input_width, const int64_t output_height,
                                 const int64_t output_width, const int32_t kernel_size_h,
                                 const int32_t kernel_size_w, const int32_t stride_h,
                                 const int32_t stride_w, const bool count_include_pad,
                                 int64_t divisor_override) {
  Avgpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_h, padding_w, n_batch,
                              n_channel, input_height, input_width, output_height, output_width,
                              kernel_size_h, kernel_size_w, stride_h, stride_w, count_include_pad,
                              divisor_override);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__ void DoCUDAAvgPool3dBackward(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
    const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
    const int32_t kernel_size_t, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_t, const int32_t stride_h, const int32_t stride_w,
    const bool count_include_pad, int64_t divisor_override) {
  Avgpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_t, padding_h, padding_w,
                              n_batch, n_channel, x_time, x_height, x_width, y_time, y_height,
                              y_width, kernel_size_t, kernel_size_h, kernel_size_w, stride_t,
                              stride_h, stride_w, count_include_pad, divisor_override);
};

template<typename T>
struct AvgPoolingKernelUtil<DeviceType::kGPU, T> {
  static void Avgpool1dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool1dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2], params_3d.count_include_pad(),
            params_3d.divisor_override());
  }

  static void Avgpool1dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool1dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
            params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2], params_3d.count_include_pad(),
            params_3d.divisor_override());
  }

  static void Avgpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool2dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
            params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
            params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
            params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
            params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool2dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
            params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
            params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
            params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
            params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool3dForward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
            params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
            params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
            params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(2),
            params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[0], params_3d.pooling_size_3d()[1],
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
            params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    DoCUDAAvgPool3dBackward<T>
        <<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0, ctx->cuda_stream()>>>(
            index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
            params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
            params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
            params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(2),
            params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
            params_3d.pooling_size_3d()[0], params_3d.pooling_size_3d()[1],
            params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
            params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_AVG_POOLING_KERNEL_UTIL, (DeviceType::kGPU),
                                 AVG_POOLING_DATA_TYPE_GPU_SEQ);

}  // namespace oneflow
