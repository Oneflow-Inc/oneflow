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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/constant_pad.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::ConstantPad> NewConstantPadPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::ConstantPadFactory>(ctx->device_type(),
                                                                        data_type);
}

template<typename Context>
std::unique_ptr<ep::primitive::ConstantPad> NewConstantPadGradPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::ConstantPadFactory>(ctx->device_type(),
                                                                        data_type);
}

auto ConstantPadPrimitiveExists() {
  return hob::make_custom("ConstantPadPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewConstantPadPrimitive(&ctx).operator bool();
  });
}

auto ConstantPadGradPrimitiveExists() {
  return hob::make_custom("ConstantPadGradPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewConstantPadGradPrimitive(&ctx).operator bool();
  });
}

}  // namespace

class PadKernel final : public OpKernel, public CudaGraphSupport {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (y->shape().NumAxes() > 0 && y->shape().elem_cnt() == 0) {
      // if output is 0-shape tensor, than do nothing and return
      return;
    }

    Scalar value;
    if (IsIntegralDataType(x->data_type())) {
      value = Scalar(ctx->Attr<int64_t>("integral_constant_value"));
    } else {
      value = Scalar(ctx->Attr<double>("floating_constant_value"));
    }

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = x->shape().NumAxes();
    CHECK_EQ(padding_before.size(), ndims);

    std::unique_ptr<ep::primitive::ConstantPad> pad_primitive = NewConstantPadPrimitive(ctx);
    CHECK(pad_primitive);

    pad_primitive->Launch(ctx->stream(), ndims, x->shape().ptr(), x->dptr(), padding_before.data(),
                          padding_after.data(), value, y->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel>().SetIsMatchedHob(ConstantPadPrimitiveExists());

class PadGradKernel final : public OpKernel, public CudaGraphSupport {
 public:
  PadGradKernel() = default;
  ~PadGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    if ((dy->shape().NumAxes() > 0 && dy->shape().elem_cnt() == 0)
        || (dx->shape().NumAxes() > 0 && dx->shape().elem_cnt() == 0)) {
      // if input/output is 0-shape tensor, than do nothing and return
      return;
    }

    std::vector<int64_t> padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    std::vector<int64_t> padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = dy->shape().NumAxes();

    for (int i = 0; i < ndims; ++i) {
      padding_before[i] = -padding_before[i];
      padding_after[i] = -padding_after[i];
    }

    std::unique_ptr<ep::primitive::ConstantPad> pad_grad_primitive =
        NewConstantPadGradPrimitive(ctx);
    CHECK(pad_grad_primitive);

    pad_grad_primitive->Launch(ctx->stream(), ndims, dy->shape().ptr(), dy->dptr(),
                               padding_before.data(), padding_after.data(), Scalar(0),
                               dx->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("pad_grad")
    .SetCreateFn<PadGradKernel>()
    .SetIsMatchedHob(ConstantPadGradPrimitiveExists());

}  // namespace user_op

}  // namespace oneflow
