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
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fused_softmax.cuh"

namespace oneflow {

namespace {

template<typename T, typename ComputeType, typename MASK, size_t num_dims>
void LaunchBroadcastForwardKernel(cudaStream_t stream, const T* x, T* y, const MASK* mask,
                                  const int64_t elem_cnt, const int64_t rows, const int64_t cols,
                                  const float fill, const float scale, const int64_t* input_dims,
                                  const int64_t* mask_dims) {
  NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims);
  NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims);
  cuda::fused_softmax::BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
  params.src_index_helper = input_index_helper;
  params.mask_index_helper = mask_index_helper;
  params.mask_dims = mask_dims;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  cuda::fused_softmax::BroadcastScaleMaskLoad<T, ComputeType, MASK, num_dims, int32_t> load(x, mask,
                                                                                            params);
  cuda::softmax::DirectStore<ComputeType, T> store(y, cols);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
}

template<typename T, typename ComputeType, typename MASK>
void LaunchElementwiseForwardKernel(cudaStream_t stream, const T* x, T* y, const MASK* mask,
                                    const int64_t rows, const int64_t cols, const float fill,
                                    const float scale) {
  cuda::fused_softmax::ElementwiseMaskSoftmaxParams params;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  cuda::fused_softmax::ElementwiseScaleMaskLoad<T, ComputeType, MASK> load(x, mask, params);
  cuda::softmax::DirectStore<ComputeType, T> store(y, cols);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
}

template<typename T, typename ComputeType, typename MASK, size_t num_dims>
void LaunchBroadcastBackwardKernel(cudaStream_t stream, const T* y, const T* dy, T* dx,
                                   const MASK* mask, const int64_t elem_cnt, const int64_t rows,
                                   const int64_t cols, const float fill, const float scale,
                                   const int64_t* input_dims, const int64_t* mask_dims) {
  NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims);
  NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims);
  cuda::fused_softmax::BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
  params.src_index_helper = input_index_helper;
  params.mask_index_helper = mask_index_helper;
  params.mask_dims = mask_dims;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, cols);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, cols);
  cuda::fused_softmax::BroadcastScaleMaskStore<ComputeType, T, MASK, num_dims, int32_t> store(
      dx, mask, params);
  OF_CUDA_CHECK((
      cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store),
                                         ComputeType>(stream, load_y, load_dy, store, rows, cols)));
}

template<typename T, typename ComputeType, typename MASK>
void LaunchElementwiseBackwardKernel(cudaStream_t stream, const T* y, const T* dy, T* dx,
                                     const MASK* mask, const int64_t rows, const int64_t cols,
                                     const float fill, const float scale) {
  cuda::fused_softmax::ElementwiseMaskSoftmaxParams params;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, cols);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, cols);
  cuda::fused_softmax::ElementwiseScaleMaskStore<ComputeType, T, MASK> store(dx, mask, params);
  OF_CUDA_CHECK((
      cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store),
                                         ComputeType>(stream, load_y, load_dy, store, rows, cols)));
}

constexpr int32_t kMaxNumDims = 5;

template<typename T, typename MASK>
class FusedScaleMaskSoftmaxKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxKernel() = default;
  ~FusedScaleMaskSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float mask_fill_value = ctx->Attr<float>("mask_fill_value");
    const float scale_value = ctx->Attr<float>("scale_value");
    const ShapeView& x_shape = x->shape_view();
    const ShapeView& mask_shape = mask->shape_view();
    CHECK_GE(x_shape.NumAxes(), 2);
    const int64_t elem_cnt = x_shape.elem_cnt();
    const int64_t cols = x_shape.At(x_shape.NumAxes() - 1);
    const int64_t rows = x_shape.Count(0, x_shape.NumAxes() - 1);
    const size_t num_input_dims = x_shape.NumAxes();
    const int64_t* input_dims = x_shape.ptr();
    const size_t num_mask_dims = mask_shape.NumAxes();
    const int64_t* mask_dims = mask_shape.ptr();
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;

    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[kMaxNumDims];
    int64_t simplified_mask_dims[kMaxNumDims];
    cuda::fused_softmax::SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims,
                                               &simplified_num_dims, simplified_input_dims,
                                               simplified_mask_dims);
    if (simplified_num_dims == 1) {
      LaunchElementwiseForwardKernel<T, ComputeType, MASK>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), x->dptr<T>(), y->mut_dptr<T>(),
          mask->dptr<MASK>(), rows, cols, mask_fill_value, scale_value);
    }
#define DEFINE_ONE_ELIF(dims)                                                               \
  else if (simplified_num_dims == dims) {                                                   \
    LaunchBroadcastForwardKernel<T, ComputeType, MASK, dims>(                               \
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), x->dptr<T>(), y->mut_dptr<T>(), \
        mask->dptr<MASK>(), elem_cnt, rows, cols, mask_fill_value, scale_value,             \
        simplified_input_dims, simplified_mask_dims);                                       \
  }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
#undef DEFINE_ONE_ELIF
    else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename MASK>
class FusedScaleMaskSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxGradKernel() = default;
  ~FusedScaleMaskSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale_value = ctx->Attr<float>("scale_value");
    const float mask_fill_value = static_cast<float>(0.0);
    const ShapeView& dy_shape = dy->shape_view();
    const ShapeView& mask_shape = mask->shape_view();
    CHECK_GE(dy_shape.NumAxes(), 2);
    const int64_t elem_cnt = dy_shape.elem_cnt();
    const int64_t cols = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t rows = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    const int64_t* input_dims = dy_shape.ptr();
    const size_t num_input_dims = dy_shape.NumAxes();
    const int64_t* mask_dims = mask_shape.ptr();
    const size_t num_mask_dims = mask_shape.NumAxes();

    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;

    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[kMaxNumDims];
    int64_t simplified_mask_dims[kMaxNumDims];
    cuda::fused_softmax::SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims,
                                               &simplified_num_dims, simplified_input_dims,
                                               simplified_mask_dims);
    if (simplified_num_dims == 1) {
      LaunchElementwiseBackwardKernel<T, ComputeType, MASK>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), y->dptr<T>(), dy->dptr<T>(),
          dx->mut_dptr<T>(), mask->dptr<MASK>(), rows, cols, mask_fill_value, scale_value);
    }
#define DEFINE_ONE_ELIF(dims)                                                                      \
  else if (simplified_num_dims == dims) {                                                          \
    LaunchBroadcastBackwardKernel<T, ComputeType, MASK, dims>(                                     \
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), y->dptr<T>(), dy->dptr<T>(),           \
        dx->mut_dptr<T>(), mask->dptr<MASK>(), elem_cnt, rows, cols, mask_fill_value, scale_value, \
        simplified_input_dims, simplified_mask_dims);                                              \
  }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
#undef DEFINE_ONE_ELIF
    else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(dtype, mask_dtype)              \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax")                                    \
      .SetCreateFn<FusedScaleMaskSoftmaxKernel<dtype, mask_dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("mask", 0) == GetDataType<mask_dtype>::value));

REGISTER_FUSED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(half, bool)
REGISTER_FUSED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(float, bool)
#undef REGISTER_FUSED_SCALE_MASK_SOFTMAX_CUDA_KERNEL

#define REGISTER_FUSED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(dtype, mask_dtype)               \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax_grad")                                \
      .SetCreateFn<FusedScaleMaskSoftmaxGradKernel<dtype, mask_dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("mask", 0) == GetDataType<mask_dtype>::value));

REGISTER_FUSED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(half, bool)
REGISTER_FUSED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(float, bool)
#undef REGISTER_FUSED_SCALE_MASK_SOFTMAX_GRAD_KERNEL

}  // namespace oneflow
