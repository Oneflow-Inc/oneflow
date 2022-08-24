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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Cast> NewCastPrimitive(Context* ctx) {
  const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::CastFactory>(ctx->device_type(), in_data_type,
                                                                 out_data_type);
}

class MutableCastOnceOpKernelState final : public OpKernelState {
 public:
  MutableCastOnceOpKernelState() : cast_once_flag_(false) {}

  void SetDone() {
    if (!cast_once_flag_) { cast_once_flag_ = true; }
  }

  bool IsDone() { return cast_once_flag_; }

 private:
  bool cast_once_flag_ = false;
};

class MutableCastOnce final : public OpKernel {
 public:
  MutableCastOnce() = default;
  ~MutableCastOnce() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    return std::make_shared<MutableCastOnceOpKernelState>();
  }

 private:
  void Compute(KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* cast_state = CHECK_NOTNULL(dynamic_cast<MutableCastOnceOpKernelState*>(state));
    if (cast_state->IsDone()) { return; }
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape_view().elem_cnt();
    CHECK_EQ(output_tensor->shape_view().elem_cnt(), elem_cnt);
    auto cast_primitive = NewCastPrimitive(ctx);
    CHECK(cast_primitive);
    cast_primitive->Launch(ctx->stream(), input_tensor->dptr(), output_tensor->mut_dptr(),
                           elem_cnt);
    cast_state->SetDone();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto CastPrimitiveExists() {
  return hob::make_custom("CastPrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewCastPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("mutable_cast_once")
    .SetCreateFn<MutableCastOnce>()
    .SetIsMatchedHob(CastPrimitiveExists() == true);

}  // namespace

}  // namespace user_op

}  // namespace oneflow
