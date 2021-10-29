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
struct GpuUnfoldTensorFunctor final {
  void operator()(DeviceCtx* ctx, const T* in_ptr, const STRIDES out_stride,
                  const STRIDES out_shape, const int32_t out_dims, const int32_t elements,
                  T* out_ptr) {
    RUN_CUDA_KERNEL((UnfoldTensorCudaKernel<T>), ctx, elements, in_ptr, out_stride, out_shape,
                    out_dims, elements, out_ptr);
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

    const ShapeView& in_shape = in->shape();
    std::vector<int32_t> out_shape;
    out_shape.resize(out->shape().NumAxes());
    for (int i = 0; i < out->shape().NumAxes(); ++i) { out_shape[i] = out->shape().At(i); }
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
    const int32_t out_size = out->shape().elem_cnt();

    STRIDES out_stride_cuda;
    for (int i = 0; i < out_dims; ++i) { out_stride_cuda.val[i] = out_stride[i]; }
    STRIDES out_shape_cuda;
    for (int i = 0; i < out_dims; ++i) { out_shape_cuda.val[i] = out_shape[i]; }

    GpuUnfoldTensorFunctor<T>()(ctx->device_ctx(), in_ptr, out_stride_cuda, out_shape_cuda,
                                out_dims, out_size, out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNFOLD_TENSOR_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("unfold_tensor")                              \
      .SetCreateFn<GpuUnfoldTensorKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU) \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))

REGISTER_UNFOLD_TENSOR_KERNEL(float);
REGISTER_UNFOLD_TENSOR_KERNEL(double);
REGISTER_UNFOLD_TENSOR_KERNEL(int32_t);
REGISTER_UNFOLD_TENSOR_KERNEL(int64_t);

}  // namespace oneflow
