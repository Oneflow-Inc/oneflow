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

class OptimFuseCastOpKernelState final : public OpKernelState {
 public:
  OptimFuseCastOpKernelState() : cast_flag(true) {}
  void set_flag_false() { cast_flag = false; }
  bool get_cast_flag() { return cast_flag; }

 private:
  bool cast_flag = true;
};

class OptimFuseCast final : public OpKernel {
 public:
  OptimFuseCast() = default;
  ~OptimFuseCast() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    return std::make_shared<OptimFuseCastOpKernelState>();
  }

 private:
  void Compute(KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* cast_state = CHECK_NOTNULL(dynamic_cast<OptimFuseCastOpKernelState*>(state));
    bool cast_flag = cast_state->get_cast_flag();
    if (!cast_flag) { return; }
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape().elem_cnt();
    CHECK_EQ(output_tensor->shape().elem_cnt(), elem_cnt);
    auto cast_primitive = NewCastPrimitive(ctx);
    CHECK(cast_primitive);
    cast_primitive->Launch(ctx->stream(), input_tensor->dptr(), output_tensor->mut_dptr(),
                           elem_cnt);
    cast_state->set_flag_false();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto CastPrimitiveExists() {
  return hob::make_custom("CastPrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewCastPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("optim_fuse_cast")
    .SetCreateFn<OptimFuseCast>()
    .SetIsMatchedHob(CastPrimitiveExists() == true);

}  // namespace

}  // namespace user_op

}  // namespace oneflow
