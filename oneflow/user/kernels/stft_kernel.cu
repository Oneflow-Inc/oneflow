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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cufft.h>
#include <cufftXt.h>
#include "oneflow/core/kernel/kernel.h"
#include "cufftplancache.h"
namespace oneflow {
namespace {}  // namespace

template<typename T, typename FFTTYPE>
__global__ void convert_complex_to_real(T* dst, const FFTTYPE* src, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[2 * i] = src[i].x;
    dst[2 * i + 1] = src[i].y;
  };
}

const double _fft_normalization_scale(const int32_t frame_length) {
  return 1.0 / std::sqrt(frame_length);
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
__global__ void convert_doublesided(FFTTYPE* dst, int32_t dims, const int n) {}

template<typename T, typename COMPLEXTYPE>
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
    const int out_elem_cnt = output->shape_view().elem_cnt() / 2;

    const T* data_in = input->dptr<T>();
    T* data_out = output->mut_dptr<T>();

    COMPLEXTYPE* out_tmp_buffer = reinterpret_cast<COMPLEXTYPE*>(tmp_buffer->mut_dptr<char>());

    int ndim = 1;
    const Stride& in_stride = {input_stride.at(2), input_stride.at(1)};
    const Stride& out_stride = {1, input_shape.At(2) / 2 + 1};
    const Shape& in_shape = {input_shape.At(2), input_shape.At(1)};
    const Shape& out_shape = {input_shape.At(2), input_shape.At(1)};
    int32_t batch = input_shape.At(1);
    int dims = 1;
    int32_t rank[dims] = {input_shape.At(2)};
    CuFFtParams params(ndim, rank, in_stride, out_stride, in_shape, out_shape, batch);
    CuFFtConfig<T, COMPLEXTYPE> config(params);

    int32_t in_offset = input->stride().at(0);
    int32_t out_offset = batch * (input_shape.At(2) / 2 + 1);

    for (int32_t i = 0; i < input_shape.At(0); i++) {
      config.excute_plan(data_in + i * in_offset, out_tmp_buffer + i * out_offset);
    }

    // TODO(yzm):support doublesided
    if (!onesided) {
      convert_doublesided<<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          out_tmp_buffer, output_shape.At(0), out_elem_cnt);
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
#define REGISTER_STFT_GPU_KERNEL(dtype, complextype)                                            \
  REGISTER_USER_KERNEL("stft")                                                                  \
      .SetCreateFn<StftGpuKernel<dtype, complextype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                       \
        const Shape& output_shape = ctx->InputShape("output", 0);                               \
        const bool return_complex = ctx->Attr<bool>("return_complex");                          \
        int32_t output_elem_cnt =                                                               \
            return_complex ? output_shape.elem_cnt() : output_shape.elem_cnt() / 2;             \
        const int64_t output_bytes = GetCudaAlignedSize(output_elem_cnt * sizeof(complextype)); \
        return output_bytes;                                                                    \
      });
REGISTER_STFT_GPU_KERNEL(float, cufftComplex)
REGISTER_STFT_GPU_KERNEL(double, cufftDoubleComplex)
}  // namespace oneflow