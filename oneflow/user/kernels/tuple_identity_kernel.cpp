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
class TupleIdentityKernel final : public user_op::OpKernel {
 public:
  TupleIdentityKernel() = default;
  ~TupleIdentityKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t in_size = ctx->user_op_conf().input_size("in");
    CHECK_EQ(ctx->user_op_conf().output_size("out"), in_size);
    for (int64_t i = 0; i < in_size; ++i) {
      const user_op::Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      user_op::Tensor* out_i = ctx->Tensor4ArgNameAndIndex("out", i);
      const DataType data_type = in_i->data_type();
      CHECK_EQ(out_i->data_type(), data_type);
      const ShapeView& shape = in_i->shape();
      CHECK_EQ(out_i->shape(), shape);
      Memcpy<device_type>(ctx->device_ctx(), out_i->mut_dptr(), in_i->dptr(),
                          shape.elem_cnt() * GetSizeOfDataType(data_type));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TUPLE_IDENTITY_KERNEL(device)    \
  REGISTER_USER_KERNEL("tuple_identity")          \
      .SetCreateFn<TupleIdentityKernel<device>>() \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device);

REGISTER_TUPLE_IDENTITY_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_TUPLE_IDENTITY_KERNEL(DeviceType::kGPU)
#endif

}  // namespace

}  // namespace oneflow
