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
#include "oneflow/user/kernels/unfold_tensor_kernel_utils.h"

namespace oneflow {

namespace {

const int32_t NDIMS = 16;
struct STRIDES {
  int32_t val[NDIMS];
};

template<typename T>
__global__ void UnfoldTensorCudaKernel(const T* in_ptr, const STRIDES out_stride,
                                       const STRIDES out_shape, const int32_t out_dims,
                                       const int32_t elements, T* out_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    int32_t offset = Offset(gid, out_stride.val, out_shape.val, out_dims - 1);
    out_ptr[gid] = in_ptr[offset];
    gid += step;
  }
}

template<typename T>
__global__ void UnfoldTensorGradCudaKernel(const T* dout_ptr, const STRIDES dout_stride,
                                           const STRIDES dout_shape, const int32_t dout_dims,
                                           const int32_t elements, T* din_ptr) {
  int32_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int32_t step = gridDim.x * blockDim.x;
  while (gid < elements) {
    int32_t offset = Offset(gid, dout_stride.val, dout_shape.val, dout_dims - 1);
    cuda::atomic::Add(&din_ptr[offset], dout_ptr[gid]);
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
struct GpuUnfoldTensorFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, const STRIDES out_stride,
                  const STRIDES out_shape, const int32_t out_dims, const int32_t elements,
                  T* out_ptr) {
    RUN_CUDA_KERNEL((UnfoldTensorCudaKernel<T>), stream, elements, in_ptr, out_stride, out_shape,
                    out_dims, elements, out_ptr);
  }
};

template<typename T>
struct GpuUnfoldTensorGradFunctor final {
  void operator()(ep::Stream* stream, const T* dout_ptr, const STRIDES dout_stride,
                  const STRIDES dout_shape, const int32_t dout_dims, const int32_t dout_elements,
                  const int32_t din_elements, T* din_ptr) {
    RUN_CUDA_KERNEL((InitPtr<T>), stream, din_elements, din_elements, din_ptr);
    RUN_CUDA_KERNEL((UnfoldTensorGradCudaKernel<T>), stream, dout_elements, dout_ptr, dout_stride,
                    dout_shape, dout_dims, dout_elements, din_ptr);
  }
};

}  // namespace

template<typename T>
class GpuUnfoldTensorKernel final : public user_op::OpKernel {
 public:
  GpuUnfoldTensorKernel() = default;
  ~GpuUnfoldTensorKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);

    const ShapeView& in_shape = in->shape_view();
    std::vector<int32_t> out_shape;
    out_shape.resize(out->shape_view().NumAxes());
    for (int i = 0; i < out->shape_view().NumAxes(); ++i) {
      out_shape[i] = out->shape_view().At(i);
    }
    const int32_t in_dims = in_shape.NumAxes();
    const int32_t out_dims = out_shape.size();
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const int32_t step = ctx->Attr<int32_t>("step");

    std::vector<int32_t> in_stride(in_dims, 1);
    for (int32_t i = in_dims - 2; i >= 0; --i) {
      in_stride[i] = in_shape.At(i + 1) * in_stride.at(i + 1);
    }

    std::vector<int32_t> out_stride(in_dims + 1);
    out_stride[in_dims] = in_dims == 0 ? 1 : in_stride[dimension];
    for (int d = 0; d < in_dims; ++d) {
      if (d == dimension) {
        out_stride[d] = step * in_stride[d];
      } else {
        out_stride[d] = in_stride[d];
      }
    }

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int32_t out_size = out->shape_view().elem_cnt();

    STRIDES out_stride_cuda;
    for (int i = 0; i < out_dims; ++i) { out_stride_cuda.val[i] = out_stride[i]; }
    STRIDES out_shape_cuda;
    for (int i = 0; i < out_dims; ++i) { out_shape_cuda.val[i] = out_shape[i]; }

    GpuUnfoldTensorFunctor<T>()(ctx->stream(), in_ptr, out_stride_cuda, out_shape_cuda, out_dims,
                                out_size, out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNFOLD_TENSOR_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("unfold_tensor")                                \
      .SetCreateFn<GpuUnfoldTensorKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))

REGISTER_UNFOLD_TENSOR_KERNEL(float);
REGISTER_UNFOLD_TENSOR_KERNEL(double);
REGISTER_UNFOLD_TENSOR_KERNEL(int32_t);
REGISTER_UNFOLD_TENSOR_KERNEL(int64_t);

template<typename T>
class GpuUnfoldTensorGradKernel final : public user_op::OpKernel {
 public:
  GpuUnfoldTensorGradKernel() = default;
  ~GpuUnfoldTensorGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* din = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const ShapeView& in_shape = in->shape_view();
    const int32_t in_dims = in_shape.NumAxes();
    std::vector<int32_t> din_stride(in_dims, 1);
    for (int32_t i = in_dims - 2; i >= 0; --i) {
      din_stride[i] = in_shape.At(i + 1) * din_stride.at(i + 1);
    }

    std::vector<int32_t> dout_shape;
    dout_shape.resize(dout->shape_view().NumAxes());
    for (int i = 0; i < dout->shape_view().NumAxes(); ++i) {
      dout_shape[i] = dout->shape_view().At(i);
    }

    const int32_t dout_dims = dout_shape.size();
    const int32_t dimension = ctx->Attr<int32_t>("dimension");
    const int32_t step = ctx->Attr<int32_t>("step");

    std::vector<int32_t> dout_stride(in_dims + 1);
    dout_stride[in_dims] = in_dims == 0 ? 1 : din_stride[dimension];
    for (int d = 0; d < in_dims; ++d) {
      if (d == dimension) {
        dout_stride[d] = step * din_stride[d];
      } else {
        dout_stride[d] = din_stride[d];
      }
    }

    STRIDES dout_stride_cuda;
    for (int i = 0; i < dout_dims; ++i) { dout_stride_cuda.val[i] = dout_stride[i]; }
    STRIDES dout_shape_cuda;
    for (int i = 0; i < dout_dims; ++i) { dout_shape_cuda.val[i] = dout_shape[i]; }

    const T* dout_ptr = dout->dptr<T>();
    T* din_ptr = din->mut_dptr<T>();
    const int32_t dout_size = dout->shape_view().elem_cnt();
    const int32_t din_size = din->shape_view().elem_cnt();

    GpuUnfoldTensorGradFunctor<T>()(ctx->stream(), dout_ptr, dout_stride_cuda, dout_shape_cuda,
                                    dout_dims, dout_size, din_size, din_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("unfold_tensor_grad")                           \
      .SetCreateFn<GpuUnfoldTensorGradKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))

REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(float);
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(double);
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(int32_t);
REGISTER_UNFOLD_TENSOR_GRAD_KERNEL(int64_t);

}  // namespace oneflow
