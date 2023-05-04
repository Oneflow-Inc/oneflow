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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/opencl/ep/cl_stream.h"

namespace oneflow {

template<typename T>
class clAddNKernel final : public user_op::OpKernel {
 public:
  clAddNKernel() = default;
  ~clAddNKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();
    const size_t count = out->shape_view().elem_cnt();

    size_t in_num = ctx->inputs().size();
    if (in_num == 0) { return; }

    const auto* in_0 = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(in_0->shape_view().elem_cnt(), count);
    CHECK_EQ(in_0->data_type(), data_type);
    Memcpy<DeviceType::kOpenCL>(ctx->stream(), out->mut_dptr(), in_0->dptr(),
                                count * GetSizeOfDataType(data_type));

    auto bcast_add = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        ctx->device_type(), ep::primitive::BinaryOp::kAdd, data_type, data_type,
        out->shape_view().NumAxes());
    CHECK(bcast_add);

    for (size_t i = 1; i < in_num; ++i) {
      const auto* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(in_i->shape_view().elem_cnt(), count);
      CHECK_EQ(in_i->data_type(), data_type);
      bcast_add->Launch(ctx->stream(), out->shape_view().NumAxes(), out->shape_view().ptr(),
                        out->dptr(), in_i->shape_view().NumAxes(), in_i->shape_view().ptr(),
                        in_i->dptr(), out->mut_dptr());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADDN_CL_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("add_n").SetCreateFn<clAddNKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kOpenCL)                             \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_ADDN_CL_KERNEL(float)

}  // namespace oneflow
