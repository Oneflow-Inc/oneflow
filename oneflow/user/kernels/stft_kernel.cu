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

#include <cuda.h>

#if CUDA_VERSION >= 11000

#include "cufft_plan_cache.h"

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

}  // namespace

template<typename IN, typename OUT>
class StftGpuKernel final : public user_op::OpKernel {
 public:
  StftGpuKernel() = default;
  ~StftGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const bool normalized = ctx->Attr<bool>("normalized");
    const bool onesided = ctx->Attr<bool>("onesided");
    const bool return_complex = ctx->Attr<bool>("return_complex");

    const ShapeView& input_shape = input->shape_view();
    const ShapeView& output_shape = output->shape_view();

    const Stride& input_stride = input->stride();
    const int out_elem_cnt =
        return_complex ? output->shape_view().elem_cnt() : output->shape_view().elem_cnt() / 2;

    const IN* data_in = input->dptr<IN>();
    IN* data_out = output->mut_dptr<IN>();
    OUT* out_tmp_buffer = reinterpret_cast<OUT*>(tmp_buffer->mut_dptr<char>());

    int32_t ndim = 1;
    int32_t n_frames = static_cast<int32_t>(input_shape.At(1));
    int32_t fft_size = static_cast<int32_t>(input_shape.At(2));
    const Stride& in_stride = {input_stride.at(2), input_stride.at(1)};
    const Stride& out_stride = {1, fft_size / 2 + 1};
    const Shape& in_shape = {fft_size, n_frames};
    const Shape& out_shape = in_shape;
    int32_t batch = n_frames;
    int32_t rank[1] = {fft_size};
    CuFFtParams params(ndim, rank, in_stride, out_stride, in_shape, out_shape, batch);
    CuFFtConfig<IN, OUT> config(params);

    int32_t in_offset = input_stride.at(0);
    int32_t out_offset = n_frames * (fft_size / 2 + 1);
    int32_t signal_groups_count = static_cast<int32_t>(input_shape.At(0));
    for (int32_t i = 0; i < signal_groups_count; i++) {
      config.excute_plan(data_in + i * in_offset, out_tmp_buffer + i * out_offset);
    }

    if (!onesided) {
      size_t last_dim_length = fft_size / 2 + 1;
      OUT* doublesided_tmp_buffer =
          reinterpret_cast<OUT*>(tmp_buffer->mut_dptr<char>()) + out_elem_cnt;
      convert_doublesided<<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          out_tmp_buffer, doublesided_tmp_buffer, last_dim_length, out_elem_cnt);
      out_tmp_buffer = doublesided_tmp_buffer;
    }

    const double normalization_scale = _fft_normalization_scale(input_shape.back());
    fft_apply_normalization<<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                              ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_tmp_buffer, normalization_scale, out_elem_cnt, normalized);

    if (!return_complex) {
      convert_complex_to_real<<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          data_out, out_tmp_buffer, out_elem_cnt);
    } else {
      // TODO(yzm):support return_complex after oneflow supports complex numbers
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_STFT_GPU_KERNEL(intype, outtype)                                           \
  REGISTER_USER_KERNEL("stft")                                                              \
      .SetCreateFn<StftGpuKernel<intype, outtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<intype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const Shape& output_shape = ctx->InputShape("output", 0);                           \
        const bool return_complex = ctx->Attr<bool>("return_complex");                      \
        const bool onesided = ctx->Attr<bool>("onesided");                                  \
        int64_t output_elem_cnt =                                                           \
            return_complex ? output_shape.elem_cnt() : output_shape.elem_cnt() / 2;         \
        const int64_t output_bytes = GetCudaAlignedSize(output_elem_cnt * sizeof(outtype)); \
        return onesided ? output_bytes : 2 * output_bytes;                                  \
      });

REGISTER_STFT_GPU_KERNEL(float, cufftComplex)
REGISTER_STFT_GPU_KERNEL(double, cufftDoubleComplex)

}  // namespace oneflow

#endif
