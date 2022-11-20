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
#include "oneflow/core/ep/include/primitive/log_softmax.h"
#include "oneflow/core/ep/include/primitive/log_softmax_backward.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::LogSoftmax> NewLogSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::LogSoftmaxFactory>(ctx->device_type(),
                                                                       data_type);
}

auto LogSoftmaxPrimitiveExists() {
  return hob::make_custom("LogSoftmaxPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewLogSoftmaxPrimitive(&ctx).operator bool();
  });
}

template<typename Context>
std::unique_ptr<ep::primitive::LogSoftmaxBackward> NewLogSoftmaxBackwardPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::LogSoftmaxBackwardFactory>(ctx->device_type(),
                                                                               data_type);
}

auto LogSoftmaxBackwardPrimitiveExists() {
  return hob::make_custom("LogSoftmaxBackwardPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewLogSoftmaxBackwardPrimitive(&ctx).operator bool();
                          });
}

class LogSoftmaxKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LogSoftmaxKernel() = default;
  ~LogSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    const ShapeView& in_shape = in->shape_view();
    const int64_t num_classes = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t num_instances = in_shape.Count(0, in_shape.NumAxes() - 1);
    std::unique_ptr<ep::primitive::LogSoftmax> primitive = NewLogSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), num_instances, num_classes, in->dptr(), prob->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class LogSoftmaxGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LogSoftmaxGradKernel() = default;
  ~LogSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = prob->shape_view().At(prob->shape_view().NumAxes() - 1);
    const int64_t num_instances = prob->shape_view().elem_cnt() / num_classes;

    std::unique_ptr<ep::primitive::LogSoftmaxBackward> primitive =
        NewLogSoftmaxBackwardPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), num_instances, num_classes, prob->dptr(), dy->dptr(),
                      dx->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("log_softmax")
    .SetCreateFn<LogSoftmaxKernel>()
    .SetIsMatchedHob(LogSoftmaxPrimitiveExists() == true);

REGISTER_USER_KERNEL("log_softmax_grad")
    .SetCreateFn<LogSoftmaxGradKernel>()
    .SetIsMatchedHob(LogSoftmaxBackwardPrimitiveExists() == true);

}  // namespace oneflow
