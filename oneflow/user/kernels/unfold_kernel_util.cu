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
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

constexpr int32_t kUnfoldMaxGpuBlockSize = 256;

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

template<typename T, typename INDEX_T, int NDIM, int SDIM>
__global__ void CudaUnfoldForward(UnfoldParams<INDEX_T, NDIM, SDIM> params, const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP_T(INDEX_T, out_offset, params.elem_cnt) {
    using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
    INDEX_T in_index[ParamType::kInputNDim] = {0};
    INDEX_T out_index[ParamType::kOutputNDim] = {0};
    params.out_index_helper.OffsetToNdIndex(out_offset, out_index);
    if (UnfoldIndexTransform<INDEX_T, NDIM, SDIM>(params, out_index, in_index)) {
      INDEX_T in_offset = params.in_index_helper.NdIndexToOffset(in_index);
      out[out_offset] = in[in_offset];
    } else {
      out[out_offset] = static_cast<T>(kUnfoldPaddingValue);
    }
  }
}

template<typename T>
__global__ void UnfoldCFirstBackward(
    const int64_t elem_cnt, const T* data_col, const int64_t depth, const int64_t height,
    const int64_t width, const int64_t kernel_d, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_depth, const int64_t pad_height, const int64_t pad_width,
    const int64_t stride_depth, const int64_t stride_height, const int64_t stride_width,
    const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
    const int64_t depth_col, const int64_t height_col, const int64_t width_col, T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    T val = static_cast<T>(0);
    const int64_t w_im = index % width + pad_width;
    const int64_t h_im = (index / width) % height + pad_height;
    const int64_t d_im = ((index / width) / height) % depth + pad_depth;
    const int64_t c_im = index / (width * height * depth);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_height + 1;
    int64_t kernel_extent_d = (kernel_d - 1) * dilation_depth + 1;

    // compute the start and end of the output
    const int64_t w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_width + 1;
    const int64_t w_col_end = ::min(w_im / stride_width + 1, width_col);
    const int64_t h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_height + 1;
    const int64_t h_col_end = ::min(h_im / stride_height + 1, height_col);
    const int64_t d_col_start =
        (d_im < kernel_extent_d) ? 0 : (d_im - kernel_extent_d) / stride_depth + 1;
    const int64_t d_col_end = ::min(d_im / stride_depth + 1, depth_col);

    for (int64_t d_col = d_col_start; d_col < d_col_end; d_col += 1) {
      for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int64_t d_k = (d_im - d_col * stride_depth);
          int64_t h_k = (h_im - h_col * stride_height);
          int64_t w_k = (w_im - w_col * stride_width);
          if (d_k % dilation_depth == 0 && h_k % dilation_height == 0
              && w_k % dilation_width == 0) {
            d_k /= dilation_depth;
            h_k /= dilation_height;
            w_k /= dilation_width;
            int64_t data_col_index =
                (((((c_im * kernel_d + d_k) * kernel_h + h_k) * kernel_w + w_k) * depth_col + d_col)
                     * height_col
                 + h_col)
                    * width_col
                + w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_im[index] = static_cast<T>(val);
  }
}

inline int32_t GetUnfoldMaxNumBlocks(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kUnfoldMaxGpuBlockSize - 1) / kUnfoldMaxGpuBlockSize, kCudaMaxBlocksNum);
}

}  // namespace

template<typename T>
class UnfoldKernelUtil<DeviceType::kGPU, T> {
 public:
  static void CFirstBackward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                             const Shape& out, const std::vector<int32_t>& kernel_size,
                             const std::vector<int32_t>& strides,
                             const std::vector<int32_t>& dilation_rate,
                             const std::vector<int32_t>& padding_before, const T* data_col,
                             T* data_im) {
    cudaMemset(data_im, T(0), in.elem_cnt() * sizeof(T));

    UnfoldCFirstBackward<T><<<GetUnfoldMaxNumBlocks(in.elem_cnt()), kUnfoldMaxGpuBlockSize, 0,
                              device_ctx->cuda_stream()>>>(
        in.elem_cnt(), data_col, in.At(2), in.At(3), in.At(4), kernel_size.at(0), kernel_size.at(1),
        kernel_size.at(2), padding_before.at(0), padding_before.at(1), padding_before.at(2),
        strides.at(0), strides.at(1), strides.at(2), dilation_rate.at(0), dilation_rate.at(1),
        dilation_rate.at(2), out_5d.At(2), out_5d.At(3), out_5d.At(4), data_im);
    OF_CUDA_CHECK(cudaGetLastError());
  }
};

template<typename T, typename INDEX_T, int NDIM, int SDIM>
struct UnfoldKernelUtilV2<DeviceType::kGPU, T, INDEX_T, NDIM, SDIM> {
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  static void Forward(DeviceCtx* ctx, const void* params, const T* input_ptr, T* output_ptr) {
    const auto* unfold_params = static_cast<const ParamType*>(params);
    CudaUnfoldForward<T, INDEX_T, NDIM, SDIM>
        <<<GetNumBlocks(unfold_params->elem_cnt), kBlockSize, 0, ctx->cuda_stream()>>>(
            *unfold_params, input_ptr, output_ptr);
  }
};

INSTANTIATE_UNFOLD_KERNEL_UTIL_FOR_DEVICE(DeviceType::kGPU)

INSTANTIATE_UNFOLD_KERNEL_UTIL(DeviceType::kGPU, float)
INSTANTIATE_UNFOLD_KERNEL_UTIL(DeviceType::kGPU, double)

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CUDA
