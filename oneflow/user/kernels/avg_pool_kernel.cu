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
#include "oneflow/user/kernels/avg_pool_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(const int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int32_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

}  // namespace

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool1dForward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                const T* src, T* dest, int32_t padding_l, const int32_t n_batch,
                                const int32_t n_channel, const int32_t x_length,
                                const int32_t kernel_size_l, const int32_t stride_l,
                                const bool count_include_pad, const int32_t divisor_override) {
  Avgpool1dForwardCompute<T>(index_helper, elem_num, src, dest, padding_l, n_batch, n_channel,
                             x_length, kernel_size_l, stride_l, count_include_pad,
                             divisor_override);
};

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool2dForward(const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num,
                                const T* src, T* dest, const int32_t padding_h,
                                const int32_t padding_w, const int32_t n_batch,
                                const int32_t n_channel, const int32_t x_height,
                                const int32_t x_width, const int32_t kernel_size_h,
                                const int32_t kernel_size_w, const int32_t stride_h,
                                const int32_t stride_w, const bool count_include_pad,
                                const int32_t divisor_override) {
  Avgpool2dForwardCompute<T>(index_helper, elem_num, src, dest, padding_h, padding_w, n_batch,
                             n_channel, x_height, x_width, kernel_size_h, kernel_size_w, stride_h,
                             stride_w, count_include_pad, divisor_override);
};

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool3dForward(const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num,
                                const T* src, T* dest, int32_t padding_t, const int32_t padding_h,
                                const int32_t padding_w, const int32_t n_batch,
                                const int32_t n_channel, const int32_t x_time,
                                const int32_t x_height, const int32_t x_width,
                                const int32_t kernel_size_t, int32_t kernel_size_h,
                                const int32_t kernel_size_w, const int32_t stride_t,
                                const int32_t stride_h, const int32_t stride_w,
                                const bool count_include_pad, const int32_t divisor_override) {
  Avgpool3dForwardCompute<T>(index_helper, elem_num, src, dest, padding_t, padding_h, padding_w,
                             n_batch, n_channel, x_time, x_height, x_width, kernel_size_t,
                             kernel_size_h, kernel_size_w, stride_t, stride_h, stride_w,
                             count_include_pad, divisor_override);
};

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool1dBackward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                 const T* src, T* dest, const int32_t padding_l,
                                 const int32_t n_batch, const int32_t n_channel,
                                 const int32_t input_length, const int32_t kernel_size_l,
                                 const int32_t stride_l, const bool count_include_pad,
                                 const int32_t divisor_override) {
  Avgpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_l, n_batch, n_channel,
                              input_length, kernel_size_l, stride_l, count_include_pad,
                              divisor_override);
};

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAAvgPool2dBackward(const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num,
                                 const T* src, T* dest, const int32_t padding_h,
                                 const int32_t padding_w, const int32_t n_batch,
                                 const int32_t n_channel, const int32_t input_height,
                                 const int32_t input_width, const int32_t kernel_size_h,
                                 const int32_t kernel_size_w, const int32_t stride_h,
                                 const int32_t stride_w, const bool count_include_pad,
                                 int32_t divisor_override) {
  Avgpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_h, padding_w, n_batch,
                              n_channel, input_height, input_width, kernel_size_h, kernel_size_w,
                              stride_h, stride_w, count_include_pad, divisor_override);
};

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__ void DoCUDAAvgPool3dBackward(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, const int32_t divisor_override) {
  Avgpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, padding_t, padding_h, padding_w,
                              n_batch, n_channel, x_time, x_height, x_width, kernel_size_t,
                              kernel_size_h, kernel_size_w, stride_t, stride_h, stride_w,
                              count_include_pad, divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoHalfAvgPool1dForward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                const half* src, half* dest, int32_t padding_l,
                                const int32_t n_batch, const int32_t n_channel,
                                const int32_t x_length, const int32_t kernel_size_l,
                                const int32_t stride_l, const bool count_include_pad,
                                const int32_t divisor_override) {
  HalfAvgpool1dForwardCompute<IDX>(index_helper, elem_num, src, dest, padding_l, n_batch, n_channel,
                                   x_length, kernel_size_l, stride_l, count_include_pad,
                                   divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoHalfAvgPool2dForward(const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num,
                                const half* src, half* dest, const int32_t padding_h,
                                const int32_t padding_w, const int32_t n_batch,
                                const int32_t n_channel, const int32_t x_height,
                                const int32_t x_width, const int32_t kernel_size_h,
                                const int32_t kernel_size_w, const int32_t stride_h,
                                const int32_t stride_w, const bool count_include_pad,
                                const int32_t divisor_override) {
  HalfAvgpool2dForwardCompute<IDX>(index_helper, elem_num, src, dest, padding_h, padding_w, n_batch,
                                   n_channel, x_height, x_width, kernel_size_h, kernel_size_w,
                                   stride_h, stride_w, count_include_pad, divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoHalfAvgPool3dForward(const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num,
                                const half* src, half* dest, int32_t padding_t,
                                const int32_t padding_h, const int32_t padding_w,
                                const int32_t n_batch, const int32_t n_channel,
                                const int32_t x_time, const int32_t x_height, const int32_t x_width,
                                const int32_t kernel_size_t, int32_t kernel_size_h,
                                const int32_t kernel_size_w, const int32_t stride_t,
                                const int32_t stride_h, const int32_t stride_w,
                                const bool count_include_pad, const int32_t divisor_override) {
  HalfAvgpool3dForwardCompute<IDX>(index_helper, elem_num, src, dest, padding_t, padding_h,
                                   padding_w, n_batch, n_channel, x_time, x_height, x_width,
                                   kernel_size_t, kernel_size_h, kernel_size_w, stride_t, stride_h,
                                   stride_w, count_include_pad, divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoHalfAvgPool1dBackward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                 const half* src, half* dest, const int32_t padding_l,
                                 const int32_t n_batch, const int32_t n_channel,
                                 const int32_t input_length, const int32_t kernel_size_l,
                                 const int32_t stride_l, const bool count_include_pad,
                                 const int32_t divisor_override) {
  HalfAvgpool1dBackwardCompute<IDX>(index_helper, elem_num, src, dest, padding_l, n_batch,
                                    n_channel, input_length, kernel_size_l, stride_l,
                                    count_include_pad, divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoHalfAvgPool2dBackward(const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num,
                                 const half* src, half* dest, const int32_t padding_h,
                                 const int32_t padding_w, const int32_t n_batch,
                                 const int32_t n_channel, const int32_t input_height,
                                 const int32_t input_width, const int32_t kernel_size_h,
                                 const int32_t kernel_size_w, const int32_t stride_h,
                                 const int32_t stride_w, const bool count_include_pad,
                                 int32_t divisor_override) {
  HalfAvgpool2dBackwardCompute<IDX>(index_helper, elem_num, src, dest, padding_h, padding_w,
                                    n_batch, n_channel, input_height, input_width, kernel_size_h,
                                    kernel_size_w, stride_h, stride_w, count_include_pad,
                                    divisor_override);
};

template<typename IDX>
__launch_bounds__(kBlockSize) __global__ void DoHalfAvgPool3dBackward(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const half* src, half* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, const int32_t divisor_override) {
  HalfAvgpool3dBackwardCompute<IDX>(index_helper, elem_num, src, dest, padding_t, padding_h,
                                    padding_w, n_batch, n_channel, x_time, x_height, x_width,
                                    kernel_size_t, kernel_size_h, kernel_size_w, stride_t, stride_h,
                                    stride_w, count_include_pad, divisor_override);
};

template<typename T, typename IDX>
struct AvgPoolKernelUtil<DeviceType::kCUDA, T, IDX> {
  static void Avgpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool1dForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                     stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool1dBackward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool2dForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                     stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool2dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool2dBackward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool3dForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                     stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    DoCUDAAvgPool3dBackward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }
};

template<typename IDX>
struct AvgPoolKernelUtil<DeviceType::kCUDA, half, IDX> {
  static void Avgpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool1dForward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool1dBackward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool2dForward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool2dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool2dBackward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool3dForward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d) {
    DoHalfAvgPool3dBackward<IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_AVG_POOL_KERNEL_UTIL, (DeviceType::kCUDA),
                                 AVG_POOL_DATA_TYPE_CUDA_SEQ, AVG_POOL_IDX_DATA_TYPE_SEQ);
template struct AvgPoolKernelUtil<DeviceType::kCUDA, half, int32_t>;
template struct AvgPoolKernelUtil<DeviceType::kCUDA, half, int64_t>;

}  // namespace oneflow
