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

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#if 1
#include <cuda.h>

#if CUDA_VERSION >= 11000

#include "cufft_plan_cache.h"
#include "oneflow/user/kernels/fft_kernel_util.h"

namespace oneflow {

namespace {

template<typename IN, typename OUT>
__global__ void convert_complex_to_real(IN* dst, const OUT* src, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[2 * i] = src[i].x;
    dst[2 * i + 1] = src[i].y;
  };
}

double _fft_normalization_scale(const int32_t frame_length) {
  return static_cast<double>(1.0 / std::sqrt(frame_length));
}

template<typename FFTTYPE>
__global__ void fft_apply_normalization(FFTTYPE* dst, const double normalization_scale, size_t n,
                                        bool IsNormalized) {
  if (!IsNormalized) { return; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[i].x *= normalization_scale;
    dst[i].y *= normalization_scale;
  };
}

// TODO(yzm):support doublesided
template<typename FFTTYPE>
__global__ void convert_doublesided(const FFTTYPE* src, FFTTYPE* dst, size_t len, size_t n) {
  size_t fact_len = 2 * len - 2;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int index_x = i / fact_len;
    int index_y = i % fact_len;
    if (index_y == 0) {
      dst[i] = src[index_x * len];
    } else if (index_y == len - 1) {
      dst[i] = src[(index_x + 1) * len - 1];
    } else if (index_y < len - 1 && index_y > 0) {
      dst[i] = src[index_x * len + index_y];
    } else {
      auto index = (index_x + 2) * len - index_y - 2;
      dst[i].x = src[index].x;
      dst[i].y = -src[index].y;
    }
  }
}

bool isCompact(const std::vector<int64_t>& strides, const std::vector<int64_t>& shape){
  if (strides.size() != shape.size()){
    return false;
  }
  Shape shape_(shape);
  Stride stride_(shape_);
  FOR_RANGE(int64_t, i, 0, strides.size()){
    if (strides[i] != stride_[i]){
      return false;
    }
  }
  return true;
}

}  // namespace

template<typename T, typename FCT_TYPE>
class FftC2CKernelUtil<DeviceType::kCUDA, T, FCT_TYPE>{
  static void FftC2CForward(ep::Stream* stream, const T* data_in, T* data_out, 
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, 
                            bool forward, const std::vector<int64_t>& dims, FCT_TYPE normalization,
                            DataType real_type){
    CuFFTParams params(input_shape, output_shape, input_stride, output_stride, 
                      dims.size(), CUFFT_EXCUTETYPE::C2C, real_type);
    CuFFTConfig config(params);
    auto& plan = config.plan();
    OF_CUFFT_CHECK(cufftSetStream(plan, stream->As<ep::CudaStream>()->cuda_stream()));
    void* workspace{};
    OF_CUDA_CHECK(cudaMalloc(&workspace, config.workspace_size()));
    OF_CUFFT_CHECK(cufftSetWorkArea(plan, workspace));

    config.excute((void*)data_in, (void*)data_out, forward);
    OF_CUDA_CHECK(cudaFree(workspace));
  }
};


template<typename IN, typename OUT>
struct FftR2CKernelUtil<DeviceType::kCUDA, IN, OUT> {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, IN normalization){
    // TO-DO:
    UNIMPLEMENTED();
  }
};

template<typename IN, typename OUT>
struct FftC2RKernelUtil<DeviceType::kCUDA, IN, OUT> {
  static void FftC2RForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride,
                            int64_t last_dim_size, const std::vector<int64_t>& dims,
                            OUT normalization){
    // TO-DO:
    UNIMPLEMENTED();
  }
};

template struct FftC2CKernelUtil<DeviceType::kCUDA, cuComplex, float>;
template struct FftC2CKernelUtil<DeviceType::kCUDA, cuDoubleComplex, double>;

}  // namespace oneflow

#endif

#endif