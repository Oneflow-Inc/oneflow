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

#if 1
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
#endif

#if 0
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
#endif
#if 0
// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
template<typename IN, typename OUT>
static void DoFFT(ep::Stream* stream, IN* in, OUT* out,
                  const Stride& in_stride, const Shape& in_shape, 
                  std::vector<int64_t>& out_sizes, std::vector<int64_t>& fft_dims, bool forward)
{
  const int64_t ndim = in_stride.size();
  const int64_t fft_ndim = fft_dims.size();
  const int64_t batch_dims = ndim - fft_ndim;


  // Permute dimensions to make batch dims come first, and this maximizes data locality
  std::vector<int64_t> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t(0));
  std::vector<bool> is_transformed_dim(ndim, false);
  for (const auto& dim : fft_dims){
    is_transformed_dim[dim] = true;
  }

  auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                        [&](int64_t d) {return !is_transformed_dim[d];});
  std::sort(dim_permute.begin(), batch_end,
            [&](int64_t a, int64_t b) { return in_stride[a] > in_stride[b]; });
  std::copy(fft_dims.begin(), fft_dims.end(), batch_end);

  // permute
  std::vector<int64_t> working_in_stride(dim_permute.size(), 0);
  std::vector<int64_t> working_in_shape(dim_permute.size(), 0);
  FOR_RANGE(int64_t, i, 0, dim_permute.size()){
    working_in_shape[i] = in_shape[dim_permute[i]];
    working_in_stride[i] = in_stride[dim_permute[i]];
  }

  std::vector<int64_t> batched_sizes(fft_ndim + 1);
  int64_t batch = 1;
  FOR_RANGE(int64_t, i, 0, working_in_shape.size() - fft_ndim){
    batch *= working_in_shape[i];
  }
  // input = input.reshape(batched_sizes)
  // maybe method:
  // `1
  // 1. judge if compact
  // 2. if compact, no need to be contiguous, else be contiguous
  // 3. change working_in_shape and working_in_stride 
  // `2
  // 1. judge if compact
  // 2. if compact, just change working_in_shape and working_in_stride
  // 3. if not compact, construct `MemcpyFactory` like reshape kernel
  if (!isCompact(/*strides=*/working_in_stride, /*shape=*/working_in_shape)){
    ToContiguousUtil<DeviceType::kCUDA, IN>(stream, )
  }
  else{

  }


}
#endif

template<typename T, typename FCT_TYPE>
class FftC2CKernelUtil<DeviceType::kCUDA, T, FCT_TYPE>{
  static void FftC2CForward(ep::Stream* stream, const T* data_in, T* data_out, 
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, 
                            bool forward, const std::vector<int64_t>& dims, FCT_TYPE normalization,
                            DataType real_type){
    CuFFTParams params(input_shape, output_shape, input_stride, output_stride, 
                      dims.size(), forward, CUFFT_EXCUTETYPE::C2C, real_type);
    CuFFTConfig config(params);
    auto& plan = config.plan();
    CUFFT_CHECK(cufftSetStream(plan, stream->As<ep::CudaStream>()->cuda_stream()));
    void* workspace{};
    OF_CUDA_CHECK(cudaMalloc(&workspace, config.workspace_size()));
    CUFFT_CHECK(cufftSetWorkArea(plan, workspace));

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