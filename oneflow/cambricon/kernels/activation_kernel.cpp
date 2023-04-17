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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace {

class MluActivationKernel final : public user_op::OpKernel {
 public:
  MluActivationKernel() = default;
  ~MluActivationKernel() = default;

  using ComputeImpl = std::function<void(user_op::KernelComputeContext* ctx)>;
  explicit MluActivationKernel(ComputeImpl compute_impl) : compute_impl_(compute_impl) {}

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override { compute_impl_(ctx); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  ComputeImpl compute_impl_;
};

void CnnlActivationForward(user_op::KernelComputeContext* ctx, const user_op::Tensor* in,
                           user_op::Tensor* out, const CnnlActivationDescriptor& activation_desc) {
  CnnlTensorDescriptor input_desc, output_desc;
  input_desc.set(in);
  output_desc.set(out);
  OF_CNNL_CHECK(cnnlActivationForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                      activation_desc.desc(), nullptr, input_desc.desc(),
                                      in->dptr(), nullptr, output_desc.desc(), out->mut_dptr()));
}

void CnnlActivationBackwardWithX(user_op::KernelComputeContext* ctx, const user_op::Tensor* in0,
                                 const user_op::Tensor* in1, user_op::Tensor* out,
                                 const CnnlActivationDescriptor& activation_desc) {
  CnnlTensorDescriptor diff_y_desc(in0), x_desc(in1), diff_x_desc(out);
  OF_CNNL_CHECK(cnnlActivationBackward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                       activation_desc.desc(),
                                       /*alpha*/ nullptr,
                                       /*y_desc*/ nullptr,
                                       /*y*/ nullptr, diff_y_desc.desc(),
                                       /*diff_y*/ in0->dptr(), x_desc.desc(),
                                       /*x. when op=relu_grad, replace x with y*/ in1->dptr(),
                                       /*beta*/ nullptr, diff_x_desc.desc(),
                                       /*diff_x*/ out->mut_dptr()));
}

inline auto BaseActivationIsMatched(const std::string& input_name) {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType(input_name, 0) == kFloat)
             || (user_op::HobDataType(input_name, 0) == kFloat16));
}

REGISTER_USER_KERNEL("relu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_RELU, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
        CnnlActivationForward(ctx, in, out, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("x"));

REGISTER_USER_KERNEL("relu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("dx", 0);
        const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_HIGH_PRECISION,
                            CNNL_NOT_PROPAGATE_NAN, /*coef*/ 0.0);
        CnnlActivationBackwardWithX(ctx, dy, y, output, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("dx"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_GELU, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
        CnnlActivationForward(ctx, in, out, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("in"));

REGISTER_USER_KERNEL("gelu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("dx", 0);
        const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_GELU, CNNL_ACTIVATION_HIGH_PRECISION,
                            CNNL_NOT_PROPAGATE_NAN, /*coef*/ 0.0);
        CnnlActivationBackwardWithX(ctx, dy, x, output, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("dx"));

REGISTER_USER_KERNEL("tanh")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
        CnnlActivationDescriptor activation_desc;
        activation_desc.set(CNNL_ACTIVATION_TANH, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
        CnnlActivationForward(ctx, in, out, activation_desc);
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("x"));

REGISTER_USER_KERNEL("fast_gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        auto* stream = ctx->stream()->As<ep::MluStream>();
        BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                          stream->device()->ncores_per_cluster());
        if (in->data_type() == DataType::kFloat16) {
          bang_fast_gelu_half_kernel(handle, in->shape_view().elem_cnt(), in->dptr(),
                                     out->mut_dptr());
        } else {
          bang_fast_gelu_kernel(handle, in->shape_view().elem_cnt(), in->dptr<float>(),
                                out->mut_dptr<float>());
        }
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("in"));

REGISTER_USER_KERNEL("fast_gelu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<MluActivationKernel>([](user_op::KernelComputeContext* ctx) {
        const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
        auto* stream = ctx->stream()->As<ep::MluStream>();
        BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                          stream->device()->ncores_per_cluster());
        if (dx->data_type() == DataType::kFloat16) {
          bang_fast_gelu_grad_half_kernel(handle, dx->shape_view().elem_cnt(), dy->dptr(),
                                          x->dptr(), dx->mut_dptr());
        } else {
          bang_fast_gelu_grad_kernel(handle, dx->shape_view().elem_cnt(), dy->dptr<float>(),
                                     x->dptr<float>(), dx->mut_dptr<float>());
        }
      });
    })
    .SetIsMatchedHob(BaseActivationIsMatched("dx"));

}  // namespace
}  // namespace oneflow
