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
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/roll_kernel_utils.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T, int Dim>
__global__ void RollCudaKernel(const T* in_ptr, const SHIFTS shifts, const SHAPE shape,
                               const STRIDE stride, const int64_t elements, T* out_ptr) {
  int32_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (global_index < elements) {
    int32_t shifted_global_index =
        getShiftedIndex<Dim>(global_index, shifts.val, shape.val, stride.val);
    out_ptr[global_index] = in_ptr[shifted_global_index];
    global_index += step;
  }
}

template<typename T, int Dim>
struct GpuRollFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const SHIFTS shifts, const SHAPE shape,
                  const STRIDE stride, const int64_t elements, T* out_ptr) {
    RollCudaKernel<T, Dim><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                             stream->As<ep::CudaStream>()->cuda_stream()>>>(
        in_ptr, shifts, shape, stride, elements, out_ptr);
  }
};

template<int Dim>
struct GpuRollFunctor<float16, Dim> final {
  void operator()(ep::Stream* stream, const float16* in_ptr, const SHIFTS shifts, const SHAPE shape,
                  const STRIDE stride, const int64_t elements, float16* out_ptr) {
    RollCudaKernel<half, Dim><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
        reinterpret_cast<const half*>(in_ptr), shifts, shape, stride, elements,
        reinterpret_cast<half*>(out_ptr));
  }
};

template<typename T>
__global__ void RollFlattenCudaKernel(const T* in_ptr, const int64_t start,
                                      const int64_t elem_count_minus_start, const int64_t elements,
                                      T* out_ptr) {
  int64_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;

  while (global_index < elements) {
    int64_t source_idx = 0;
    if (global_index >= elem_count_minus_start) {
      source_idx = global_index - elem_count_minus_start;
    } else {
      source_idx = global_index + start;
    }
    out_ptr[global_index] = in_ptr[source_idx];

    global_index += step;
  }
}

template<typename T>
struct GpuRollFlattenFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const int64_t start,
                  const int64_t elem_count_minus_start, const int64_t elements, T* out_ptr) {
    RollFlattenCudaKernel<T><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                               stream->As<ep::CudaStream>()->cuda_stream()>>>(
        in_ptr, start, elem_count_minus_start, elements, out_ptr);
  }
};

template<>
void GpuRollFlattenFunctor<float16>::operator()(ep::Stream* stream, const float16* in_ptr,
                                                const int64_t start,
                                                const int64_t elem_count_minus_start,
                                                const int64_t elements, float16* out_ptr) {
  RollFlattenCudaKernel<half><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(in_ptr), start, elem_count_minus_start, elements,
      reinterpret_cast<half*>(out_ptr));
}

template<typename T>
__global__ void Roll1DimCudaKernel(const T* in_ptr, const int32_t stride_x_size,
                                   const int32_t stride, const int32_t size_minus_start,
                                   const int32_t size_minus_start_x_stride,
                                   const int32_t start_x_stride, const int64_t elements,
                                   T* out_ptr) {
  int32_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;

  while (global_index < elements) {
    // roll dim idx is the index of linear_index along the rolling dimension.
    int32_t roll_dim_idx = global_index % stride_x_size / stride;
    // index into the source data to find appropriate value.
    int32_t source_idx = 0;
    if (roll_dim_idx >= size_minus_start) {
      source_idx = global_index - size_minus_start_x_stride;
    } else {
      source_idx = global_index + start_x_stride;
    }
    out_ptr[global_index] = in_ptr[source_idx];

    global_index += step;
  }
}

template<typename T>
struct GpuRoll1DimFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const int32_t stride_x_size,
                  const int32_t stride, const int32_t size_minus_start,
                  const int32_t size_minus_start_x_stride, const int32_t start_x_stride,
                  const int64_t elements, T* out_ptr) {
    Roll1DimCudaKernel<T><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                            stream->As<ep::CudaStream>()->cuda_stream()>>>(
        in_ptr, stride_x_size, stride, size_minus_start, size_minus_start_x_stride, start_x_stride,
        elements, out_ptr);
  }
};

template<>
void GpuRoll1DimFunctor<float16>::operator()(ep::Stream* stream, const float16* in_ptr,
                                             const int32_t stride_x_size, const int32_t stride,
                                             const int32_t size_minus_start,
                                             const int32_t size_minus_start_x_stride,
                                             const int32_t start_x_stride, const int64_t elements,
                                             float16* out_ptr) {
  Roll1DimCudaKernel<half><<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0,
                             stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(in_ptr), stride_x_size, stride, size_minus_start,
      size_minus_start_x_stride, start_x_stride, elements, reinterpret_cast<half*>(out_ptr));
}

}  // namespace

template<typename T>
class GpuRollKernel final : public user_op::OpKernel {
 public:
  GpuRollKernel() = default;
  ~GpuRollKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& shifts = ctx->Attr<std::vector<int32_t>>("shifts");
    const std::vector<int32_t>& dims = ctx->Attr<std::vector<int32_t>>("dims");

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int64_t elem_count = out->shape_view().elem_cnt();

    if (dims[0] == -1) {
      // NOTE(Liang Depeng): Borrow the implementation of pytorch and simplify to 1d array case.
      int64_t start = (elem_count - shifts[0]) % elem_count;
      if (start < 0) start = start + elem_count;
      const int64_t elem_count_minus_start = elem_count - start;
      GpuRollFlattenFunctor<T>()(ctx->stream(), in_ptr, start, elem_count_minus_start, elem_count,
                                 out_ptr);
    } else {
      SHAPE new_shape{};
      SHIFTS new_shifts{};
      int32_t num_axes = 0;
      computeParams(in->shape_view(), shifts, dims, new_shifts.val, new_shape.val, &num_axes);

      STRIDE stride{};
      initStride(stride, new_shape, num_axes);

      if (dims.size() == 1) {
        // NOTE(Liang Depeng): Borrow the implementation of pytorch
        const int32_t size = new_shape.val[dims[0]];
        int32_t start = (size - new_shifts.val[dims[0]]) % size;
        // Behavior of % is different in C++ vs Python for negative numbers. This
        // corrects the difference.
        if (start < 0) start = start + size;

        const int32_t stride_x_size = stride.val[dims[0]] * size;
        const int32_t size_minus_start = size - start;
        const int32_t size_minus_start_x_stride = size_minus_start * stride.val[dims[0]];
        const int32_t start_x_stride = start * stride.val[dims[0]];

        GpuRoll1DimFunctor<T>()(ctx->stream(), in_ptr, stride_x_size, stride.val[dims[0]],
                                size_minus_start, size_minus_start_x_stride, start_x_stride,
                                elem_count, out_ptr);

      } else {
        transformShifts(new_shifts.val, new_shape.val, num_axes);
        switch (num_axes) {
          case 1:
            GpuRollFunctor<T, 1>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 2:
            GpuRollFunctor<T, 2>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 3:
            GpuRollFunctor<T, 3>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 4:
            GpuRollFunctor<T, 4>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 5:
            GpuRollFunctor<T, 5>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 6:
            GpuRollFunctor<T, 6>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 7:
            GpuRollFunctor<T, 7>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 8:
            GpuRollFunctor<T, 8>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 9:
            GpuRollFunctor<T, 9>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride, elem_count,
                                   out_ptr);
            break;
          case 10:
            GpuRollFunctor<T, 10>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 11:
            GpuRollFunctor<T, 11>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 12:
            GpuRollFunctor<T, 12>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 13:
            GpuRollFunctor<T, 13>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 14:
            GpuRollFunctor<T, 14>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 15:
            GpuRollFunctor<T, 15>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          case 16:
            GpuRollFunctor<T, 16>()(ctx->stream(), in_ptr, new_shifts, new_shape, stride,
                                    elem_count, out_ptr);
            break;
          default: break;
        }
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ROLL_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("roll").SetCreateFn<GpuRollKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                               \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_ROLL_KERNEL(float);
REGISTER_ROLL_KERNEL(double);
REGISTER_ROLL_KERNEL(float16);
REGISTER_ROLL_KERNEL(bool);
REGISTER_ROLL_KERNEL(uint8_t);
REGISTER_ROLL_KERNEL(int8_t);
REGISTER_ROLL_KERNEL(int32_t);
REGISTER_ROLL_KERNEL(int64_t);

}  // namespace oneflow
