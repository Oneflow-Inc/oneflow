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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace {

using namespace oneflow;

class MluBroadcastBinaryKernelBase : public user_op::OpKernel {
 public:
  MluBroadcastBinaryKernelBase() = default;
  ~MluBroadcastBinaryKernelBase() = default;

  virtual void ComputeImpl(user_op::KernelComputeContext* ctx,
                           const std::array<cnnlTensorDescriptor_t, 2>& input_descs,
                           const std::array<const void*, 2>& input_dptrs,
                           const cnnlTensorDescriptor_t& z_desc, void* z_dtr) const = 0;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    std::array<const void*, 2> input_dptrs = {x->dptr(), y->dptr()};

    std::array<CnnlTensorDescriptor, 3> descs{x, y, z};

    std::array<cnnlTensorDescriptor_t, 2> input_descs{descs.at(0).desc(), descs.at(1).desc()};

    const auto z_desc = descs.at(2).desc();

    void* z_dtr = z->mut_dptr();

    ComputeImpl(ctx, input_descs, input_dptrs, z_desc, z_dtr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

namespace oneflow {

template<typename T>
class MluBroadcastAddN final : public MluBroadcastBinaryKernelBase {
 public:
  void ComputeImpl(user_op::KernelComputeContext* ctx,
                   const std::array<cnnlTensorDescriptor_t, 2>& input_descs,
                   const std::array<const void*, 2>& input_dptrs,
                   const cnnlTensorDescriptor_t& z_desc, void* z_dtr) const override {
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAddNWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                           input_descs.data(), input_descs.size(), z_desc,
                                           &workspace_size));
    CnnlWorkspace cnnl_workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlAddN_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), input_descs.data(),
                              input_dptrs.data(), input_descs.size(), z_desc, z_dtr,
                              cnnl_workspace.dptr(), workspace_size));
  }
};

template<typename T>
class MluBroadcastMul final : public MluBroadcastBinaryKernelBase {
 public:
  void ComputeImpl(user_op::KernelComputeContext* ctx,
                   const std::array<cnnlTensorDescriptor_t, 2>& input_descs,
                   const std::array<const void*, 2>& input_dptrs,
                   const cnnlTensorDescriptor_t& z_desc, void* z_dtr) const override {
    const auto cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    OF_CNNL_CHECK(cnnlExpand(cnnl_handle, input_descs.at(0), input_dptrs.at(0), z_desc, z_dtr));
    size_t workspace_size = 0;
    const auto a_desc = input_descs.at(1);
    OF_CNNL_CHECK(cnnlGetAxWorkspaceSize(cnnl_handle, a_desc, z_desc, &workspace_size));
    CnnlWorkspace cnnl_workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlAx_v2(cnnl_handle, a_desc, input_dptrs.at(1), z_desc, z_dtr,
                            cnnl_workspace.dptr(), workspace_size));
  }
};

template<typename T>
class MluBroadcastDiv final : public MluBroadcastBinaryKernelBase {
 public:
  void ComputeImpl(user_op::KernelComputeContext* ctx,
                   const std::array<cnnlTensorDescriptor_t, 2>& input_descs,
                   const std::array<const void*, 2>& input_dptrs,
                   const cnnlTensorDescriptor_t& z_desc, void* z_dtr) const override {
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetDivWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                          input_descs.at(0), input_descs.at(1), z_desc,
                                          &workspace_size));
    CnnlWorkspace cnnl_workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlDiv_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                             CNNL_COMPUTATION_HIGH_PRECISION, input_descs.at(0), input_dptrs.at(0),
                             input_descs.at(1), input_dptrs.at(1), cnnl_workspace.dptr(),
                             workspace_size, z_desc, z_dtr));
  }
};

#define REGISTER_BROADCAST_OP_MLU_KERNEL(name, op, dtype)              \
  REGISTER_USER_KERNEL(name).SetCreateFn<op<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                   \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_add", MluBroadcastAddN, float)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_add", MluBroadcastAddN, float16)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_add", MluBroadcastAddN, int8_t)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_add", MluBroadcastAddN, uint8_t)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_add", MluBroadcastAddN, int32_t)

REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_mul", MluBroadcastMul, float)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_mul", MluBroadcastMul, float16)

REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_div", MluBroadcastDiv, float)
REGISTER_BROADCAST_OP_MLU_KERNEL("broadcast_div", MluBroadcastDiv, float16)

REGISTER_BROADCAST_OP_MLU_KERNEL("div", MluBroadcastDiv, float)
REGISTER_BROADCAST_OP_MLU_KERNEL("div", MluBroadcastDiv, float16)

}  // namespace oneflow