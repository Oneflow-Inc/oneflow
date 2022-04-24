/*
Memcpyright 2020 The OneFlow Authors. All rights reserved.

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
#include "oneflow/core/ep/include/primitive/memcpy.h"
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

template<typename Context>
std::unique_ptr<ep::primitive::Memcpy> NewMemcpyPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->device_type(), ep::primitive::MemcpyKind::kDtoD);
}

class OptimFuseCastOpKernelState final : public OpKernelState {
    public: 
        OptimFuseCastOpKernelState(): cast_cnt(0){
        }
        void set_one(){
            cast_cnt = 1; 
        }
        int64_t get_cast_cnt() {
            return cast_cnt; 
        }
    private: 
        int64_t cast_cnt = 0; 
}; 

class OptimFuseCast final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  OptimFuseCast() = default;
  ~OptimFuseCast() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    return std::make_shared<OptimFuseCastOpKernelState>();
  }

 private:
  void Compute(KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape().elem_cnt();
    auto* cast_state = CHECK_NOTNULL(dynamic_cast<OptimFuseCastOpKernelState*>(state));

    int64_t cast_cnt = cast_state->get_cast_cnt(); 

    CHECK_EQ(output_tensor->shape().elem_cnt(), elem_cnt);
    if (input_tensor->data_type() == output_tensor->data_type()
        && input_tensor->dptr() == output_tensor->dptr()) {
      return;
    }
    if(cast_cnt == 0){
        auto cast_primitive = NewCastPrimitive(ctx);
        CHECK(cast_primitive);
        cast_primitive->Launch(ctx->stream(), input_tensor->dptr(), output_tensor->mut_dptr(), elem_cnt);
        cast_state->set_one(); 
    } else {
        auto memcpy_primitive = NewMemcpyPrimitive(ctx);
        CHECK(memcpy_primitive);
        memcpy_primitive->Launch(ctx->stream(), output_tensor->mut_dptr(), input_tensor->dptr(), elem_cnt);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto CastPrimitiveExists() {
  return hob::make_custom("CastPrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewCastPrimitive(&ctx).operator bool();
  });
}

auto MemcpyNdPrimitiveExists() {
  return hob::make_custom("MemcpyNdPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewMemcpyPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("optim_fuse_cast")
    .SetCreateFn<OptimFuseCast>()
    .SetIsMatchedHob(CastPrimitiveExists() == true && MemcpyNdPrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.InputDType("in", 0) == ctx.Attr<DataType>("dtype")) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      }
      return Maybe<void>::Ok();
    });


}  // namespace

}  // namespace user_op

}  // namespace oneflow
