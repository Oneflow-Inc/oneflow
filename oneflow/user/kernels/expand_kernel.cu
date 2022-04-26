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
#include "oneflow/user/kernels/expand_kernel_utils.h"

namespace oneflow {

namespace {

const int32_t NDIMS = 16;
struct STRIDES {
  int32_t val[NDIMS];
};

template<typename T>
__global__ void ExpandCudaKernel(const T* in_ptr, const STRIDES in_stride,
                                 const STRIDES expand_stride, const int32_t dims,
                                 const int32_t elements, T* out_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    int32_t offset = OffsetToNdIndexToOffset(gid, in_stride.val, expand_stride.val, dims);
    out_ptr[gid] = in_ptr[offset];
    gid += step;
  }
}

template<typename T>
__global__ void ExpandGradCudaKernel(const T* out_diff_ptr, const STRIDES out_stride,
                                     const STRIDES expand_stride, const int32_t dims,
                                     const int32_t elements, T* in_diff_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    int32_t offset = OffsetToNdIndexToOffset(gid, out_stride.val, expand_stride.val, dims);
    cuda::atomic::Add(&in_diff_ptr[offset], out_diff_ptr[gid]);
    gid += step;
  }
}

template<typename T>
__global__ void InitPtr(const int32_t elements, T* ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    ptr[gid] = static_cast<T>(0);
    gid += step;
  }
}

template<typename T>
struct GpuExpandFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const STRIDES in_stride,
                  const STRIDES expand_stride, const int32_t dims, const int32_t elements,
                  T* out_ptr) {
    RUN_CUDA_KERNEL((ExpandCudaKernel<T>), stream, elements, in_ptr, in_stride, expand_stride, dims,
                    elements, out_ptr);
  }
};

template<>
void GpuExpandFunctor<float16>::operator()(ep::Stream* stream, const float16* in_ptr,
                                           const STRIDES in_stride, const STRIDES expand_stride,
                                           const int32_t dims, const int32_t elements,
                                           float16* out_ptr) {
  RUN_CUDA_KERNEL((ExpandCudaKernel<half>), stream, elements, reinterpret_cast<const half*>(in_ptr),
                  in_stride, expand_stride, dims, elements, reinterpret_cast<half*>(out_ptr));
}

template<typename T>
struct GpuExpandGradFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const STRIDES in_stride,
                  const STRIDES expand_stride, const int32_t dims, const int32_t elements,
                  const int32_t out_elements, T* out_ptr) {
    RUN_CUDA_KERNEL((InitPtr<T>), stream, out_elements, out_elements, out_ptr);
    RUN_CUDA_KERNEL((ExpandGradCudaKernel<T>), stream, elements, in_ptr, in_stride, expand_stride,
                    dims, elements, out_ptr);
  }
};

template<>
void GpuExpandGradFunctor<float16>::operator()(ep::Stream* stream, const float16* in_ptr,
                                               const STRIDES in_stride, const STRIDES expand_stride,
                                               const int32_t dims, const int32_t elements,
                                               const int32_t out_elements, float16* out_ptr) {
  RUN_CUDA_KERNEL((InitPtr<half>), stream, out_elements, out_elements,
                  reinterpret_cast<half*>(out_ptr));
  RUN_CUDA_KERNEL((ExpandGradCudaKernel<half>), stream, elements,
                  reinterpret_cast<const half*>(in_ptr), in_stride, expand_stride, dims, elements,
                  reinterpret_cast<half*>(out_ptr));
}

}  // namespace

template<typename T>
class GpuExpandKernel final : public user_op::OpKernel {
 public:
  GpuExpandKernel() = default;
  ~GpuExpandKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& logical_expand_shape =
        ctx->Attr<std::vector<int32_t>>("logical_expand_shape");
    if (std::any_of(logical_expand_shape.begin(), logical_expand_shape.end(),
                    [](int32_t dim_size) { return dim_size == 0; })) {
      return;
    }
    std::vector<int32_t> in_shape;
    in_shape.resize(in->shape().NumAxes());
    for (int i = 0; i < in->shape().NumAxes(); ++i) { in_shape[i] = in->shape().At(i); }

    std::vector<int32_t> out_shape;
    std::vector<int32_t> stride;
    CHECK_JUST(getOutShapeAndStrideForFp(in_shape, logical_expand_shape, out_shape, stride));

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int32_t out_dims = out->shape().NumAxes();
    const int32_t out_size = out->shape().elem_cnt();

    STRIDES expand_stride;
    for (int i = 0; i < out_dims; ++i) { expand_stride.val[i] = stride[i]; }
    STRIDES out_stride;
    InitStride(out_stride.val, out_shape.data(), out_dims);
    GpuExpandFunctor<T>()(ctx->stream(), in_ptr, out_stride, expand_stride, out_dims, out_size,
                          out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EXPAND_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("expand").SetCreateFn<GpuExpandKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_EXPAND_KERNEL(float);
REGISTER_EXPAND_KERNEL(double);
REGISTER_EXPAND_KERNEL(float16);
REGISTER_EXPAND_KERNEL(bool);
REGISTER_EXPAND_KERNEL(uint8_t);
REGISTER_EXPAND_KERNEL(int8_t);
REGISTER_EXPAND_KERNEL(int32_t);
REGISTER_EXPAND_KERNEL(int64_t);

template<typename T>
class GpuExpandGradKernel final : public user_op::OpKernel {
 public:
  GpuExpandGradKernel() = default;
  ~GpuExpandGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& logical_out_shape =
        ctx->Attr<std::vector<int32_t>>("logical_out_shape");
    const std::vector<int32_t>& logical_expand_shape =
        ctx->Attr<std::vector<int32_t>>("logical_expand_shape");

    std::vector<int32_t> in_shape;
    in_shape.resize(in->shape().NumAxes());
    for (int i = 0; i < in->shape().NumAxes(); ++i) { in_shape[i] = in->shape().At(i); }
    std::vector<int32_t> out_shape;
    std::vector<int32_t> stride;
    CHECK_JUST(getOutShapeAndStrideForBp(logical_out_shape, logical_expand_shape, in_shape,
                                         out_shape, stride));

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    const int32_t in_dims = in->shape().NumAxes();
    const int32_t in_size = in->shape().elem_cnt();
    const int32_t out_size = out->shape().elem_cnt();

    STRIDES expand_stride;
    for (int i = 0; i < in_dims; ++i) { expand_stride.val[i] = stride[i]; }
    STRIDES in_stride;
    InitStride(in_stride.val, in_shape.data(), in_dims);

    GpuExpandGradFunctor<T>()(ctx->stream(), in_ptr, in_stride, expand_stride, in_dims, in_size,
                              out_size, out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EXPAND_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("expand_grad")                                  \
      .SetCreateFn<GpuExpandGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_EXPAND_GRAD_KERNEL(float);
REGISTER_EXPAND_GRAD_KERNEL(double);
REGISTER_EXPAND_GRAD_KERNEL(float16);
REGISTER_EXPAND_GRAD_KERNEL(int32_t);
REGISTER_EXPAND_GRAD_KERNEL(int64_t);

}  // namespace oneflow
