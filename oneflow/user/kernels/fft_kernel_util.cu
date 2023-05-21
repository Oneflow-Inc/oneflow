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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"

#if CUDA_VERSION >= 11000
#include "oneflow/user/kernels/fft_kernel_util.h"
#include "cufft_plan_cache.h"

namespace oneflow {

namespace {
template<typename FFTTYPE>
__global__ void fft_apply_normalization(FFTTYPE* dst, const double normalization_scale, size_t n,
                                        bool IsNormalized) {
  if (!IsNormalized) { return; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[i].x *= normalization_scale;
    dst[i].y *= normalization_scale;
  };
}

struct FillConjSymmetricParams {
  int64_t last_dim;
  int64_t elem_count;
  int64_t ndim;
  oneflow::NdIndexStrideOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> helper;
  int64_t last_dim_size;
  int64_t last_dim_half;

  FillConjSymmetricParams() = default;
  FillConjSymmetricParams(const Shape& shape, const Stride& strides, int64_t last_dim_,
                          int64_t elemcnt)
      : last_dim(last_dim_),
        elem_count(elemcnt),
        ndim(strides.size()),
        helper(strides.data(), ndim) {
    CHECK_OR_THROW(strides.size() == shape.size());
    last_dim_size = shape[last_dim];
    last_dim_half = last_dim_size / 2;
  }
};

}  // namespace

template<typename T>
__global__ void _conj_symmetry_cuda(T* data_out, FillConjSymmetricParams param) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, offset, param.elem_count) {
    int64_t ndim = param.ndim;
    int64_t indices[SHAPE_MAX_AXIS_SIZE];
    param.helper.OffsetToNdIndex(offset, indices, ndim);
    if (indices[param.last_dim] <= param.last_dim_half) { continue; }
    int64_t cur_last_dim_index = indices[param.last_dim];
    // get symmetric
    indices[param.last_dim] = param.last_dim_size - cur_last_dim_index;
    int64_t symmetric_offset = param.helper.NdIndexToOffset(indices, ndim);

    // conj
    data_out[offset] = T{data_out[symmetric_offset].x, -data_out[symmetric_offset].y};
  }
}

template<typename T>
struct FillConjSymmetryUtil<DeviceType::kCUDA, T> {
  static void FillConjSymmetryForward(ep::Stream* stream, T* data_out, const Shape& shape,
                                      const Stride& strides, const int64_t last_dim,
                                      int64_t elem_count) {
    FillConjSymmetricParams param(shape, strides, last_dim, elem_count);
    _conj_symmetry_cuda<T><<<BlocksNum4ThreadsNum(elem_count), kCudaThreadsNumPerBlock, 0,
                             stream->As<ep::CudaStream>()->cuda_stream()>>>(data_out, param);
  }
};

template<typename IN, typename OUT>
__global__ void _convert_to_double_sized(const IN* in, OUT* dst, size_t len, size_t n) {
  size_t fact_len = 2 * len - 2;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int index_x = i / fact_len;
    int index_y = i % fact_len;
    if (index_y == 0) {
      dst[i] = in[index_x * len];
    } else if (index_y == len - 1) {
      dst[i] = in[(index_x + 1) * len - 1];
    } else if (index_y < len - 1 && index_y > 0) {
      dst[i] = in[index_x * len + index_y];
    } else {
      auto index = (index_x + 2) * len - index_y - 2;
      dst[i].x = in[index].x;
      dst[i].y = -in[index].y;
    }
  }
}

template<typename IN, typename OUT>
__global__ void _convert_complex_to_real(const IN* in, OUT* out, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[2 * i] = in[i].x;
    out[2 * i + 1] = in[i].y;
  };
}

template<typename real_type, typename complex_type>
struct ComplexConvertUtil<DeviceType::kCUDA, real_type, complex_type> {
  static void ConvertToDoubleSized(ep::Stream* stream, const complex_type* in, complex_type* dst,
                                   size_t len, size_t n) {
    _convert_to_double_sized<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                               stream->As<ep::CudaStream>()->cuda_stream()>>>(in, dst, len, n);
  }
  static void ConvertComplexToReal(ep::Stream* stream, const complex_type* in, real_type* out,
                                   size_t n) {
    _convert_complex_to_real<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                               stream->As<ep::CudaStream>()->cuda_stream()>>>(in, out, n);
  }
};

template<typename dtype_in, typename dtype_out>
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

    const dtype_in* data_in = input->dptr<dtype_in>();
    dtype_in* data_out = output->mut_dptr<dtype_in>();
    dtype_out* out_tmp_buffer = reinterpret_cast<dtype_out*>(tmp_buffer->mut_dptr<char>());

    int64_t ndim = 1;
    int64_t batch = static_cast<int32_t>(input_shape.At(1));
    int64_t fft_size = static_cast<int32_t>(input_shape.At(2));
    int64_t rank[1] = {fft_size};
    const Stride& in_stride = {input_stride.at(1), input_stride.at(2)};
    const Shape& in_shape = {batch, fft_size};
    const Shape& out_shape = {batch, fft_size / 2 + 1};
    Stride out_stride = Stride(out_shape);
    CuFFTParams params(in_shape, out_shape, in_stride, out_stride, ndim, CUFFT_EXCUTETYPE::R2C,
                       input->data_type());
    CuFFTConfig config(params);
    auto& plan = config.plan();
    OF_CUFFT_CHECK(cufftSetStream(plan, ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    void* workspace{};
    OF_CUDA_CHECK(cudaMalloc(&workspace, config.workspace_size()));
    OF_CUFFT_CHECK(cufftSetWorkArea(plan, workspace));

    int64_t in_offset = input_stride.at(0);
    int64_t out_offset =
        std::accumulate(out_shape.begin(), out_shape.end(), 0, std::multiplies<int64_t>());
    int64_t signal_groups_count = static_cast<int64_t>(input_shape.At(0));
    for (int64_t i = 0; i < signal_groups_count; i++) {
      config.excute((void*)(data_in + i * in_offset), (void*)(out_tmp_buffer + i * out_offset),
                    /*forward=*/true);
    }
    OF_CUDA_CHECK(cudaFree(workspace));

    if (!onesided) {
      size_t last_dim_length = fft_size / 2 + 1;
      dtype_out* doublesided_tmp_buffer =
          reinterpret_cast<dtype_out*>(tmp_buffer->mut_dptr<char>()) + out_elem_cnt;
      ComplexConvertUtil<DeviceType::kCUDA, dtype_in, dtype_out>::ConvertToDoubleSized(
          ctx->stream(), out_tmp_buffer, doublesided_tmp_buffer, last_dim_length, out_elem_cnt);
      out_tmp_buffer = doublesided_tmp_buffer;
    }

    const double normalization_scale =
        _fft_normalization_scale<double>(input_shape.back(), normalized);
    fft_apply_normalization<<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                              ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_tmp_buffer, normalization_scale, out_elem_cnt, normalized);

    if (!return_complex) {
      ComplexConvertUtil<DeviceType::kCUDA, dtype_in, dtype_out>::ConvertComplexToReal(
          ctx->stream(), out_tmp_buffer, data_out, out_elem_cnt);
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

template<typename T, typename FCT_TYPE>
class FftC2CKernelUtil<DeviceType::kCUDA, T, FCT_TYPE> {
  static void FftC2CForward(ep::Stream* stream, const T* data_in, T* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, FCT_TYPE normalization,
                            DataType real_type) {
    // NOTE: before calling `FftC2CKernelUtil<DeviceType::kCUDA, T, FCT_TYPE>`, input must be
    // batched out already
    CuFFTParams params(input_shape, output_shape, input_stride, output_stride, dims.size(),
                       CUFFT_EXCUTETYPE::C2C, real_type);
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
                            const std::vector<int64_t>& dims, IN normalization,
                            DataType real_type) {
    // NOTE: before calling `FftR2CKernelUtil<DeviceType::kCUDA, IN, OUT>`, input must be batched
    // out already
    CuFFTParams params(input_shape, output_shape, input_stride, output_stride, dims.size(),
                       CUFFT_EXCUTETYPE::R2C, real_type);
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
struct FftC2RKernelUtil<DeviceType::kCUDA, IN, OUT> {
  static void FftC2RForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            int64_t last_dim_size, const std::vector<int64_t>& dims,
                            OUT normalization, DataType real_type) {
    // NOTE: before calling `FftC2RKernelUtil<DeviceType::kCUDA, IN, OUT>`, input must be batched
    // out already
    CuFFTParams params(input_shape, output_shape, input_stride, output_stride, dims.size(),
                       CUFFT_EXCUTETYPE::C2R, real_type);
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

template struct FillConjSymmetryUtil<DeviceType::kCUDA, cuComplex>;
template struct FillConjSymmetryUtil<DeviceType::kCUDA, cuDoubleComplex>;

template struct ComplexConvertUtil<DeviceType::kCUDA, float, cuComplex>;
template struct ComplexConvertUtil<DeviceType::kCUDA, double, cuDoubleComplex>;

template struct FftC2CKernelUtil<DeviceType::kCUDA, cuComplex, /*FCT_TYPE=*/float>;
template struct FftC2CKernelUtil<DeviceType::kCUDA, cuDoubleComplex, /*FCT_TYPE=*/double>;

template struct FftR2CKernelUtil<DeviceType::kCUDA, float, cuComplex>;
template struct FftR2CKernelUtil<DeviceType::kCUDA, double, cuDoubleComplex>;

template struct FftC2RKernelUtil<DeviceType::kCUDA, cuComplex, float>;
template struct FftC2RKernelUtil<DeviceType::kCUDA, cuDoubleComplex, double>;
}  // namespace oneflow

#endif  // CUDA_VERSION >= 11000
