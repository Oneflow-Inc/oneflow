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
#include "oneflow/core/primitive/include/softmax.h"
#include "oneflow/core/primitive/include/softmax_backward.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<primitive::Softmax> NewSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return primitive::NewPrimitive<primitive::SoftmaxFactory>(ctx->device_type(), data_type);
}

hob::HobContextGetter<user_op::KernelRegContext, bool> SoftmaxPrimitiveExists() {
  return user_op::HobCtxGetter<bool>("SoftmaxPrimitiveExists",
                                     [](const user_op::KernelRegContext& ctx) {
                                       return NewSoftmaxPrimitive(&ctx).operator bool();
                                     });
}

template<typename Context>
std::unique_ptr<primitive::SoftmaxBackward> NewSoftmaxBackwardPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->data_type();
  return primitive::NewPrimitive<primitive::SoftmaxBackwardFactory>(ctx->device_type(), data_type);
}

hob::HobContextGetter<user_op::KernelRegContext, bool> SoftmaxBackwardPrimitiveExists() {
  return user_op::HobCtxGetter<bool>("SoftmaxBackwardPrimitiveExists",
                                     [](const user_op::KernelRegContext& ctx) {
                                       return NewSoftmaxBackwardPrimitive(&ctx).operator bool();
                                     });
}

}  // namespace

class SoftmaxKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);
    std::unique_ptr<primitive::Softmax> primitive = NewSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream_ctx(), rows, cols, in->dptr(), out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("softmax").SetCreateFn<SoftmaxKernel>().SetIsMatchedHob(
    SoftmaxPrimitiveExists() == true);

class SoftmaxGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t num_instances = y->shape().elem_cnt() / num_classes;

    std::unique_ptr<primitive::SoftmaxBackward> primitive = NewSoftmaxBackwardPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream_ctx(), num_instances, num_classes, y->dptr(), dy->dptr(),
                      dx->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("softmax_grad")
    .SetCreateFn<SoftmaxGradKernel>()
    .SetIsMatchedHob(SoftmaxBackwardPrimitiveExists() == true);

}  // namespace oneflow
