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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/user/utils/unfold_util.h"
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

class UnfoldOpKernelState final : public user_op::OpKernelState {
 public:
  UnfoldOpKernelState(ParamsUnfold3D params_3d) : params_3d(params_3d) {}
  const ParamsUnfold3D& GetParams3D() { return params_3d; }

  static std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(
      user_op::KernelInitContext* ctx, const int32_t& dim) {
    if (2 != dim) { UNIMPLEMENTED(); }
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    ParamsUnfold3D params_3d =
        ParamsUnfold3D(dim, x_shape, data_format, padding, padding_before, padding_after,
                       kernel_size, strides, dilation_rate, ceil_mode);
    std::shared_ptr<UnfoldOpKernelState> state(new UnfoldOpKernelState(params_3d));
    return std::move(state);
  }

  ParamsUnfold3D params_3d;
};

template<DeviceType device_type, typename T>
class UnfoldKernelImpl {
 public:
  static void FWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* unfold_state = dynamic_cast<UnfoldOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const ParamsUnfold3D& params_3d = unfold_state->GetParams3D();

    if (data_format == "channels_first") {
      UnfoldKernelUtil<device_type, T>::CFirstForward(
          ctx->device_ctx(), params_3d.GetXShape5D(), params_3d.GetYShape5D(),
          params_3d.GetYShape(), params_3d.kernel_size_3d(), params_3d.strides_3d(),
          params_3d.dilation_rate_3d(), params_3d.padding_before_3d(), x->dptr<T>(),
          y->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }

  static void BWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* unfold_state = dynamic_cast<UnfoldOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const ParamsUnfold3D& params_3d = unfold_state->GetParams3D();

    if (data_format == "channels_first") {
      UnfoldKernelUtil<device_type, T>::CFirstBackward(
          ctx->device_ctx(), params_3d.GetXShape5D(), params_3d.GetYShape5D(),
          params_3d.GetYShape(), params_3d.kernel_size_3d(), params_3d.strides_3d(),
          params_3d.dilation_rate_3d(), params_3d.padding_before_3d(), dy->dptr<T>(),
          dx->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }
};

}  // namespace

template<size_t NDims, DeviceType device_type, typename T>
class UnfoldKernel final : public user_op::OpKernel {
 public:
  UnfoldKernel() = default;
  ~UnfoldKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldOpKernelState::DoCreateOpKernelState(ctx, NDims);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldKernelImpl<device_type, T>::FWCompute(ctx, state);
  };
};

template<size_t NDims, DeviceType device_type, typename T>
class UnfoldGradKernel final : public user_op::OpKernel {
 public:
  UnfoldGradKernel() = default;
  ~UnfoldGradKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldOpKernelState::DoCreateOpKernelState(ctx, NDims);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldKernelImpl<device_type, T>::BWCompute(ctx, state);
  };
};

#define REGISTER_UNFOLD_KERNEL_NDIMS(dim, device, dtype)                               \
  REGISTER_USER_KERNEL("unfold_" + std::to_string(dim) + "d")                          \
      .SetCreateFn<UnfoldKernel<dim, device, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_" + std::to_string(dim) + "d_grad")                     \
      .SetCreateFn<UnfoldGradKernel<dim, device, dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kCPU, float)
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kGPU, float)
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kGPU, double)
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
