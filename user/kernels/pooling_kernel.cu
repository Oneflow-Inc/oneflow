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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize << 2;

int GetMinThreadNum(int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

template<typename T>
__device__ __inline__ void Maxpool2dForwardComputeCLast(
    const NdIndexOffsetHelper<int64_t, 4>& index_helper, int64_t elem_num, const T* src, T* dest,
    int64_t* indice_ptr, const int32_t padding_h, const int32_t padding_w, const int64_t n_batch,
    const int64_t n_channel, const int64_t x_height, const int64_t x_width, const int64_t y_height,
    const int64_t y_width, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_h, const int32_t stride_w, const int32_t dilation_h,
    const int32_t dilation_w) {
  int64_t n, h, w, c;
  CUDA_1D_KERNEL_LOOP(num, elem_num) {
    index_helper.OffsetToNdIndex(num, n, h, w, c);

    const int64_t x_start_idx = n * n_channel * x_width * x_height;
    const int64_t y_start_idx = n * n_channel * y_height * y_width;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    const int64_t hend = (hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height
                             ? (hstart + (kernel_size_h - 1) * dilation_h + 1)
                             : x_height;
    const int64_t wend = (wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width
                             ? (wstart + (kernel_size_w - 1) * dilation_w + 1)
                             : x_width;

    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }
    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    int64_t max_index = hstart * x_width + wstart;
    int64_t src_idx = 0;
    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();

    for (int64_t i = hstart; i < hend; i++) {
      for (int64_t j = wstart; j < wend; j++) {
        const int64_t window_idx = i * x_width * n_channel + j * n_channel + c;
        const int64_t search_idx = x_start_idx + window_idx;
        T val = src[search_idx];
        if (val > max_value || detail::numerics<T>::isnan(val)) {
          max_value = val;
          max_index = window_idx;
          src_idx = search_idx;
        }
      }
    }
    const int64_t out_idx = y_start_idx + h * y_width * n_channel + w * n_channel + c;
    dest[out_idx] = src[src_idx];
    indice_ptr[out_idx] = max_index;
  }
}

}  // namespace

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
    void DoCUDAMaxPool2dForwardCFirst(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                      int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                      int32_t padding_h, int32_t padding_w, int64_t n_batch,
                                      int64_t n_channel, int64_t x_height, int64_t x_width,
                                      int64_t y_height, int64_t y_width, int32_t kernel_size_h,
                                      int32_t kernel_size_w, int32_t stride_h, int32_t stride_w,
                                      int32_t dilation_h, int32_t dilation_w) {
  Maxpool2dForwardComputeCFirst<T>(index_helper, elem_num, src, dest, indice_ptr, padding_h,
                                   padding_w, n_batch, n_channel, x_height, x_width, y_height,
                                   y_width, kernel_size_h, kernel_size_w, stride_h, stride_w,
                                   dilation_h, dilation_w);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool2dForwardCLast(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                     int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                     int32_t padding_h, int32_t padding_w, int64_t n_batch,
                                     int64_t n_channel, int64_t x_height, int64_t x_width,
                                     int64_t y_height, int64_t y_width, int32_t kernel_size_h,
                                     int32_t kernel_size_w, int32_t stride_h, int32_t stride_w,
                                     int32_t dilation_h, int32_t dilation_w) {
  Maxpool2dForwardComputeCLast<T>(index_helper, elem_num, src, dest, indice_ptr, padding_h,
                                  padding_w, n_batch, n_channel, x_height, x_width, y_height,
                                  y_width, kernel_size_h, kernel_size_w, stride_h, stride_w,
                                  dilation_h, dilation_w);
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
    void DoCUDAMaxPool2dBackwardCFirst(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                       const int64_t elem_num, const T* src, T* dest,
                                       const int64_t* indice_ptr, const int64_t n_batch,
                                       const int64_t n_channel, const int64_t src_height,
                                       const int64_t src_width, const int64_t dst_height,
                                       const int64_t dst_width) {
  Maxpool2dBackwardComputeCFirst<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch,
                                    n_channel, src_height, src_width, dst_height, dst_width);
};

template<typename T>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxPool2dBackwardCLast(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                      const int64_t elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr, const int64_t n_batch,
                                      const int64_t n_channel, const int64_t src_height,
                                      const int64_t src_width, const int64_t dst_height,
                                      const int64_t dst_width) {
  Maxpool2dBackwardComputeCLast<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch,
                                   n_channel, src_height, src_width, dst_height, dst_width);
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
struct PoolingKernelUtil<DeviceType::kCUDA, T> {
  static void Maxpool1dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool1dForward<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(4),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[2]);
  }

  static void Maxpool1dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool1dBackward<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                 stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCFirst(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool2dForwardCFirst<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCFirst(ep::Stream* stream,
                                      const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                      const int64_t elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr,
                                      const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool2dBackwardCFirst<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                       stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCLast(ep::Stream* stream,
                                    const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                    const int64_t elem_num, const T* src, T* dest,
                                    int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool2dForwardCLast<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                     stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCLast(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     const int64_t* indice_ptr,
                                     const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool2dBackwardCLast<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool3dForward<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[0],
        params_3d.padding()[1], params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[0],
        params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.dilation_3d()[0],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool3dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    DoCUDAMaxPool3dBackward<T><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                 stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kCUDA),
                                 POOLING_DATA_TYPE_CUDA_SEQ);

}  // namespace oneflow
#endif  // WITH_CUDA
