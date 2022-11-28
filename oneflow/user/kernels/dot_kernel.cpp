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
#include "oneflow/core/ep/include/primitive/matmul.h"

namespace oneflow {

namespace {

using namespace ep::primitive;

template<typename Context>
std::unique_ptr<Matmul> NewMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<MatmulFactory>(ctx->device_type(), data_type,
                                                    BlasTransposeType::N, BlasTransposeType::N);
}

auto MatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMatmulPrimitive(&ctx).operator bool();
  });
}

class DotKernel final : public user_op::OpKernel {
 public:
  DotKernel() = default;
  ~DotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = x->shape_view().elem_cnt();
    auto primitive = NewMatmulPrimitive(ctx);

    primitive->Launch(ctx->stream(), 1, 1, n, 1, x->dptr(), y->dptr(), 0, out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("dot").SetCreateFn<DotKernel>().SetIsMatchedHob(MatmulPrimitiveExists()
                                                                     == true);

}  // namespace

}  // namespace oneflow
