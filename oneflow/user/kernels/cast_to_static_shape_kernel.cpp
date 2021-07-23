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

namespace oneflow {

namespace {

template<DeviceType device_type>
class CastToStaticShapeKernel final : public user_op::OpKernel {
 public:
  CastToStaticShapeKernel() = default;
  ~CastToStaticShapeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Shape& input_static_shape = ctx->TensorDesc4ArgNameAndIndex("input", 0)->shape();
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    CHECK(input_tensor->shape() == ShapeView(input_static_shape));
    CHECK_EQ(output_tensor->shape(), input_tensor->shape());
    size_t output_tensor_size =
        output_tensor->shape().elem_cnt() * GetSizeOfDataType(output_tensor->data_type());
    Memcpy<device_type>(ctx->device_ctx(), output_tensor->mut_dptr(), input_tensor->dptr(),
                        output_tensor_size);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_CAST_TO_STATIC_SHAPE_KERNEL(device)                                            \
  REGISTER_USER_KERNEL("cast_to_static_shape")                                                  \
      .SetCreateFn<CastToStaticShapeKernel<device>>()                                           \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device)                                       \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("output", 0, "input", 0, false));                \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_CAST_TO_STATIC_SHAPE_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_CAST_TO_STATIC_SHAPE_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
