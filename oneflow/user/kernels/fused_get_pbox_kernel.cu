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
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(ctx->device_type(), 2);
}

}  // namespace

template<typename T>
class FusedGetPboxKernel final : public user_op::OpKernel {
 public:
  FusedGetPboxKernel() = default;
  ~FusedGetPboxKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* pxy = ctx->Tensor4ArgNameAndIndex("pxy", 0);
    const user_op::Tensor* pwh = ctx->Tensor4ArgNameAndIndex("pwh", 0);
    const user_op::Tensor* anchors = ctx->Tensor4ArgNameAndIndex("anchors", 0);
    const T* anchors_ptr = anchors->dptr<T>();

    user_op::Tensor* pbox = ctx->Tensor4ArgNameAndIndex("pbox", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* pbox_ptr = pbox->mut_dptr<T>();
    T* tmp_buffer_ptr = tmp_buffer->mut_dptr<T>();

    const int64_t elem_cnt = pxy->shape_view().elem_cnt();
    const int64_t batch_size = pxy->shape_view().At(0);

    // pxy
    const ShapeView& pxy_shape = pxy->shape_view();
    const int64_t cols = pxy_shape.At(pxy_shape.NumAxes() - 1);
    const int64_t rows = pxy_shape.Count(0, pxy_shape.NumAxes() - 1);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> pxy_load(pxy->dptr<T>(), cols);
    cuda::softmax::DirectStore<ComputeType, T> tmp_store(tmp_buffer->mut_dptr<T>(), cols);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(pxy_load), decltype(tmp_store), ComputeType>(
      ctx->stream()->As<ep::CudaStream>()->cuda_stream(), pxy_load, tmp_store, rows, cols)));
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      tmp_buffer_ptr[i] = tmp_buffer_ptr[i] * static_cast<T>(2.0) - static_cast<T>(0.5);
    }

    // concat
    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive);

    DimVector dst_shape = {batch_size, 2 * cols};
    DimVector dst_pos_vec = {0, 0};
    DimVector src_shape = {pxy_shape.At(0), pxy_shape.Count(1)};
    DimVector src_pos_vec = {0, 0};
    DimVector extent_vec = {pxy_shape.At(0), pxy_shape.Count(1)};
    primitive->Launch(ctx->stream(), tmp_buffer->data_type(), 2, pbox_ptr, dst_shape.data(),
                      dst_pos_vec.data(), tmp_buffer_ptr, src_shape.data(), src_pos_vec.data(),
                      extent_vec.data());

    // pwh
    cuda::softmax::DirectLoad<T, ComputeType> pwh_load(pwh->dptr<T>(), cols);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(pwh_load), decltype(tmp_store), ComputeType>(
      ctx->stream()->As<ep::CudaStream>()->cuda_stream(), pwh_load, tmp_store, rows, cols)));
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      const T tmp_buffer_i_2 = (tmp_buffer_ptr[i] * static_cast<T>(2.0));
      tmp_buffer_ptr[i] = tmp_buffer_i_2 * tmp_buffer_i_2 * anchors_ptr[i];
    }

    // concat 
    DimVector dst_pos_vec2 = {0, cols};
    primitive->Launch(ctx->stream(), tmp_buffer->data_type(), 2, pbox_ptr, dst_shape.data(),
                      dst_pos_vec2.data(), tmp_buffer_ptr, src_shape.data(), src_pos_vec.data(),
                      extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_PBOX_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("fused_get_pbox")                              \
      .SetCreateFn<FusedGetPboxKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDataType("pbox", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {   \
        const Shape& shape = ctx->InputShape("pxy", 0);               \
        size_t tmp_buffer_size = shape.elem_cnt() * sizeof(dtype);    \
        return tmp_buffer_size;                                       \
      });

REGISTER_FUSED_GET_PBOX_KERNEL(float)
REGISTER_FUSED_GET_PBOX_KERNEL(double)
REGISTER_FUSED_GET_PBOX_KERNEL(half)

template<typename T>
class FusedGetPboxGradKernel final : public user_op::OpKernel {
 public:
  FusedGetPboxGradKernel() = default;
  ~FusedGetPboxGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* pxy = ctx->Tensor4ArgNameAndIndex("pxy", 0);
    const user_op::Tensor* pwh = ctx->Tensor4ArgNameAndIndex("pwh", 0);
    const user_op::Tensor* anchors = ctx->Tensor4ArgNameAndIndex("anchors", 0);
    const user_op::Tensor* pbox_diff = ctx->Tensor4ArgNameAndIndex("pbox_diff", 0);
    const T* pxy_ptr = pxy->dptr<T>();
    const T* pwh_ptr = pwh->dptr<T>();
    const T* anchors_ptr = anchors->dptr<T>();
    const T* pbox_diff_ptr = pbox_diff->dptr<T>();

    user_op::Tensor* pxy_diff = ctx->Tensor4ArgNameAndIndex("pxy_diff", 0);
    user_op::Tensor* pwh_diff = ctx->Tensor4ArgNameAndIndex("pwh_diff", 0);
    user_op::Tensor* anchors_diff = ctx->Tensor4ArgNameAndIndex("anchors_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* pxy_diff_ptr = pxy_diff->mut_dptr<T>();
    T* pwh_diff_ptr = pwh_diff->mut_dptr<T>();
    T* anchors_diff_ptr = anchors_diff->mut_dptr<T>();
    T* tmp_buffer_ptr = tmp_buffer->mut_dptr<T>();

    const int64_t elem_cnt = pxy->shape_view().elem_cnt();
    const int64_t batch_size = pxy->shape_view().At(0);

    // pxy
    const ShapeView& pxy_shape = pxy->shape_view();
    const int64_t cols = pxy_shape.At(pxy_shape.NumAxes() - 1);
    const int64_t rows = pxy_shape.Count(0, pxy_shape.NumAxes() - 1);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> pxy_load(pxy_ptr, cols);
    cuda::softmax::DirectLoad<T, ComputeType> pbox_diff_load(pbox_diff_ptr, cols);
    cuda::softmax::DirectStore<ComputeType, T> tmp_store(tmp_buffer->mut_dptr<T>(), cols);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(pxy_load), decltype(pbox_diff_load), decltype(tmp_store), ComputeType>(
      ctx->stream()->As<ep::CudaStream>()->cuda_stream(), pxy_load, pbox_diff_load, tmp_store, rows, cols)));
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      pxy_diff_ptr[i] = tmp_buffer_ptr[i] * static_cast<T>(2.0);
    }

    // pwh
    cuda::softmax::DirectLoad<T, ComputeType> pwh_load(pwh_ptr, cols);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(pwh_load), decltype(pbox_diff_load), decltype(tmp_store), ComputeType>(
      ctx->stream()->As<ep::CudaStream>()->cuda_stream(), pwh_load, pbox_diff_load, tmp_store, rows, cols)));
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      pwh_diff_ptr[i] = anchors_ptr[i] * tmp_buffer_ptr[i] * static_cast<T>(8.0);
    }

    // anchors
    CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
      anchors_diff_ptr[i] = static_cast<T>(4.0) * tmp_buffer_ptr[i] * tmp_buffer_ptr[i];
    }

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_PBOX_GRAD_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("fused_get_pbox_grad")                              \
      .SetCreateFn<FusedGetPboxGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDataType("pxy_diff", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {        \
        const Shape& shape = ctx->InputShape("pxy", 0);                    \
        size_t tmp_buffer_size = shape.elem_cnt() * sizeof(dtype);         \
        return tmp_buffer_size;                                            \
      });

REGISTER_FUSED_GET_PBOX_GRAD_KERNEL(float)
REGISTER_FUSED_GET_PBOX_GRAD_KERNEL(double)
REGISTER_FUSED_GET_PBOX_GRAD_KERNEL(half)

}  // namespace oneflow
