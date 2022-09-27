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
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/softmax_backward.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Softmax> NewGumbelSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::SoftmaxFactory>(ctx->device_type(), data_type);
}

auto GumbelSoftmaxPrimitiveExists() {
  return hob::make_custom("GumbelSoftmaxPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewGumbelSoftmaxPrimitive(&ctx).operator bool();
  });
}

}  //  namespace

class GumbelSoftmaxKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GumbelSoftmaxKernel() = default;
  ~GumbelSoftmaxKernel() override = default;

 private:
  // using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const ShapeView& in_shape = in->shape_view();
    const int64_t num_classes = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t num_instances = in_shape.Count(0, in_shape.NumAxes() - 1);

    // TODO(hujiakui): Gumbel Softmax Forward，這裡是直接搬來一個softmax測試一下前向是不是加上了算子
    std::unique_ptr<ep::primitive::Softmax> primitive = NewGumbelSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), num_instances, num_classes, in->dptr(), out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gumbel_softmax")
    .SetCreateFn<GumbelSoftmaxKernel>()
    .SetIsMatchedHob(GumbelSoftmaxPrimitiveExists() == true);

}  //  namespace oneflow
