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

namespace oneflow {

namespace {

template<typename T>
__global__ void RollCudaKernel(const T* in_ptr, const SHIFTS shifts, const SHAPE shape,
                               const STRIDE stride, const int32_t dims, const int64_t elements,
                               T* out_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    int32_t offset = getShiftedOffset(gid, shifts.val, shape.val, stride.val, dims);
    out_ptr[gid] = in_ptr[offset];
    gid += step;
  }
}

template<typename T>
struct GpuRollFunctor final {
  void operator()(DeviceCtx* ctx, const T* in_ptr, const SHIFTS shifts, const SHAPE shape,
                  const STRIDE stride, const int32_t dims, const int64_t elements, T* out_ptr) {
    RollCudaKernel<T>
        <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            in_ptr, shifts, shape, stride, dims, elements, out_ptr);
  }
};

template<>
void GpuRollFunctor<float16>::operator()(DeviceCtx* ctx, const float16* in_ptr, const SHIFTS shifts,
                                         const SHAPE shape, const STRIDE stride, const int32_t dims,
                                         const int64_t elements, float16* out_ptr) {
  RollCudaKernel<half>
      <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(in_ptr), shifts, shape, stride, dims, elements,
          reinterpret_cast<half*>(out_ptr));
}

template<typename T>
__global__ void RollFlattenCudaKernel(const T* in_ptr, const int32_t shifts, const int64_t elements,
                                      T* out_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;

  while (gid < elements) {
    int32_t shifted_idx = (gid - shifts) % elements;
    if (shifted_idx < 0) shifted_idx = shifted_idx + elements;
    out_ptr[gid] = in_ptr[shifted_idx];
    gid += step;
  }
}

template<typename T>
struct GpuRollFlattenFunctor final {
  void operator()(DeviceCtx* ctx, const T* in_ptr, const int32_t shifts, const int64_t elements,
                  T* out_ptr) {
    RollFlattenCudaKernel<T>
        <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            in_ptr, shifts, elements, out_ptr);
  }
};

template<>
void GpuRollFlattenFunctor<float16>::operator()(DeviceCtx* ctx, const float16* in_ptr,
                                                const int32_t shifts, const int64_t elements,
                                                float16* out_ptr) {
  RollFlattenCudaKernel<half>
      <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(in_ptr), shifts, elements,
          reinterpret_cast<half*>(out_ptr));
}

template<typename T>
__global__ void Roll1DimCudaKernel(const T* in_ptr, const int32_t start, const int32_t size,
                                   const int32_t stride, const int64_t elements, T* out_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;

  while (gid < elements) {
    // roll dim idx is the index of linear_index along the rolling dimension.
    int32_t roll_dim_idx = gid % (stride * size) / stride;
    // index into the source data to find appropriate value.
    int32_t source_idx = 0;
    if (roll_dim_idx >= (size - start)) {
      source_idx = gid - ((size - start) * stride);
    } else {
      source_idx = gid + (start * stride);
    }
    out_ptr[gid] = in_ptr[source_idx];

    gid += step;
  }
}

template<typename T>
struct GpuRoll1DimFunctor final {
  void operator()(DeviceCtx* ctx, const T* in_ptr, const int32_t start, const int32_t size,
                  const int32_t stride, const int64_t elements, T* out_ptr) {
    Roll1DimCudaKernel<T>
        <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            in_ptr, start, size, stride, elements, out_ptr);
  }
};

template<>
void GpuRoll1DimFunctor<float16>::operator()(DeviceCtx* ctx, const float16* in_ptr,
                                             const int32_t start, const int32_t size,
                                             const int32_t stride, const int64_t elements,
                                             float16* out_ptr) {
  Roll1DimCudaKernel<half>
      <<<BlocksNum4ThreadsNum(elements), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(in_ptr), start, size, stride, elements,
          reinterpret_cast<half*>(out_ptr));
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
    const int64_t elem_count = out->shape().elem_cnt();

    if (dims[0] == -1) {
      GpuRollFlattenFunctor<T>()(ctx->device_ctx(), in_ptr, shifts[0], elem_count, out_ptr);
    } else {
      SHAPE new_shape{};
      SHIFTS new_shifts{};
      int32_t num_axes = 0;
      computeParams(in->shape(), shifts, dims, new_shifts.val, new_shape.val, &num_axes);

      STRIDE stride{};
      initStride(stride, new_shape, num_axes);

      if (dims.size() == 1) {
        // NOTE(Liang Depeng): borrow the implementation of pytorch
        const int32_t size = new_shape.val[dims[0]];
        int32_t start = (size - new_shifts.val[dims[0]]) % size;
        // Behavior of % is different in C++ vs Python for negative numbers. This
        // corrects the difference.
        if (start < 0) start = start + size;

        GpuRoll1DimFunctor<T>()(ctx->device_ctx(), in_ptr, start, size, stride.val[dims[0]],
                                elem_count, out_ptr);

      } else {
        GpuRollFunctor<T>()(ctx->device_ctx(), in_ptr, new_shifts, new_shape, stride, num_axes,
                            elem_count, out_ptr);
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ROLL_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("roll").SetCreateFn<GpuRollKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == DeviceType::kGPU)                                 \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_ROLL_KERNEL(float);
REGISTER_ROLL_KERNEL(double);
REGISTER_ROLL_KERNEL(float16);
REGISTER_ROLL_KERNEL(uint8_t);
REGISTER_ROLL_KERNEL(int8_t);
REGISTER_ROLL_KERNEL(int32_t);
REGISTER_ROLL_KERNEL(int64_t);

}  // namespace oneflow
