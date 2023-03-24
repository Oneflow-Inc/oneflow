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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

std::vector<int32_t> ComputeReducedAxis(const ShapeView& broadcasted_shape,
                                        const ShapeView& src_shape) {
  std::vector<int32_t> reduce_axis;
  int64_t expand_dim = broadcasted_shape.NumAxes() - src_shape.NumAxes();
  for (int64_t i = 0; i < broadcasted_shape.NumAxes(); ++i) {
    if (i < expand_dim) {
      reduce_axis.emplace_back(i);
      continue;
    }
    if (broadcasted_shape.At(i) != 1 && src_shape.At(i - expand_dim) == 1) {
      reduce_axis.emplace_back(i);
    }
  }
  return reduce_axis;
}

template<typename T>
class BroadcastDivGradKernel final : public user_op::OpKernel {
 public:
  BroadcastDivGradKernel() = default;
  ~BroadcastDivGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* z_tensor = ctx->Tensor4ArgNameAndIndex("z", 0);
    const user_op::Tensor* dz_tensor = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);

    size_t tmp_workspace_size =
        z_tensor->shape_view().elem_cnt() * GetSizeOfDataType(z_tensor->data_type());
    CnnlWorkspace tmp_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_workspace_size);
    void* tmp_ptr = tmp_cnnl_workspace.dptr();

    auto bcast_div = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        ctx->device_type(), ep::primitive::BinaryOp::kDiv, z_tensor->data_type(),
        z_tensor->data_type(), z_tensor->shape_view().NumAxes());
    CHECK(bcast_div);
    bcast_div->Launch(ctx->stream(), z_tensor->shape_view().NumAxes(), z_tensor->shape_view().ptr(),
                      z_tensor->dptr(), y_tensor->shape_view().NumAxes(),
                      y_tensor->shape_view().ptr(), y_tensor->dptr(), tmp_ptr);

    auto bcast_mul = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        ctx->device_type(), ep::primitive::BinaryOp::kMul, dz_tensor->data_type(),
        dz_tensor->data_type(), dz_tensor->shape_view().NumAxes());
    CHECK(bcast_mul);
    bcast_mul->Launch(ctx->stream(), dz_tensor->shape_view().NumAxes(),
                      dz_tensor->shape_view().ptr(), tmp_ptr, dz_tensor->shape_view().NumAxes(),
                      dz_tensor->shape_view().ptr(), dz_tensor->dptr(), tmp_ptr);

    CnnlTensorDescriptor tmp_desc, dy_desc;
    tmp_desc.set_reduce(dz_tensor);
    dy_desc.set_reduce(dy_tensor);

    const auto& axis = ComputeReducedAxis(dz_tensor->shape_view(), dy_tensor->shape_view());

    if (axis.empty()) {
      // axis being empty means that dz has the same shape as dy, no reduction needed.
      Memcpy<DeviceType::kMLU>(
          ctx->stream(), dy_tensor->mut_dptr(), tmp_ptr,
          dy_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dy_tensor->data_type()));
    } else {
      auto reduce_mode = CNNL_REDUCE_ADD;
      auto reduce_indices = CNNL_REDUCE_NO_INDICES;
      auto reduce_indices_type = CNNL_32BIT_INDICES;
      CnnlReduceDescriptor reduce_desc;
      auto cnnl_dtype = ConvertToCnnlDataType(dz_tensor->data_type());
      if (axis.size() == dz_tensor->shape_view().NumAxes()) {
        std::vector<int32_t> full_reduce(1, -1);
        std::vector<int32_t> fake_size(dz_tensor->shape_view().NumAxes(), 1);
        reduce_desc.set(cnnl_dtype, full_reduce, reduce_mode, reduce_indices, reduce_indices_type);
        dy_desc.set(fake_size.size(), fake_size.data(), cnnl_dtype, CNNL_LAYOUT_NCHW);
      } else {
        reduce_desc.set(cnnl_dtype, axis, reduce_mode, reduce_indices, reduce_indices_type);
      }

      size_t tmp_dy_workspace_size = 0;
      OF_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                                 tmp_desc.desc(), dy_desc.desc(),
                                                 reduce_desc.mut_desc(), &tmp_dy_workspace_size));
      CnnlWorkspace tmp_dy_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dy_workspace_size);

      OF_CNNL_CHECK(cnnlReduce(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                               reduce_desc.desc(), tmp_dy_workspace.dptr(), tmp_dy_workspace_size,
                               nullptr, tmp_desc.desc(), tmp_ptr, 0, nullptr, nullptr,
                               dy_desc.desc(), dy_tensor->mut_dptr()));
    }

    OF_CNNL_CHECK(cnnlNegTensor(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), dy_desc.desc(),
                                dy_tensor->dptr(), dy_desc.desc(), dy_tensor->mut_dptr()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_MLU_BROADCAST_DIV_GRAD_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("broadcast_div_grad")                          \
      .SetCreateFn<BroadcastDivGradKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MLU_BROADCAST_DIV_GRAD_KERNEL(float)
REGISTER_MLU_BROADCAST_DIV_GRAD_KERNEL(float16)

}  // namespace oneflow
