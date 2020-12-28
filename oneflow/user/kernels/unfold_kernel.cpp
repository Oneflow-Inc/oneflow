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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename INDEX_T, int NDIM, int SDIM>
class UnfoldOpKernelState : public OpKernelState {
 public:
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  UnfoldOpKernelState(const ShapeView& input_shape, const std::vector<int32_t>& kernel_size,
                      const std::vector<int32_t>& padding_before,
                      const std::vector<int32_t>& padding_after, const std::vector<int32_t>& stride,
                      const std::vector<int32_t>& dilation)
      : params_(input_shape.At(0), input_shape.At(ParamType::kInputChannelDim),
                input_shape.ptr() + SDIM, kernel_size.data(), padding_before.data(),
                padding_after.data(), stride.data(), dilation.data()) {}
  const ParamType& params() const { return params_; }

 private:
  ParamType params_;
};

template<typename INDEX_T, int NDIM, int SDIM>
std::shared_ptr<OpKernelState> CreateUnfoldOpKernelState(const ShapeView& input_shape,
                                                         const std::vector<int32_t>& kernel_size,
                                                         const std::vector<int32_t>& padding_before,
                                                         const std::vector<int32_t>& padding_after,
                                                         const std::vector<int32_t>& stride,
                                                         const std::vector<int32_t>& dilation) {
  return std::make_shared<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>>(
      input_shape, kernel_size, padding_before, padding_after, stride, dilation);
}

template<typename INDEX_T, int NDIM, int SDIM>
const void* GetUnfoldParams(OpKernelState* state) {
  auto* unfold_state = dynamic_cast<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>*>(state);
  CHECK_NOTNULL(unfold_state);
  return static_cast<const void*>(&unfold_state->params());
}

#define SWITCH_ENTRY(func_name, itype, ndim, sdim) func_name<itype, ndim, sdim>
#define DEFINE_UNFOLD_SWITCH_FUNC(ret_type, func_name)                                 \
  DEFINE_STATIC_SWITCH_FUNC(                                                           \
      ret_type, func_name, SWITCH_ENTRY, MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ), \
      MAKE_NDIM_CTRV_SEQ(SPATIAL_NDIM_SEQ), MAKE_NDIM_CTRV_SEQ(SPATIAL_DIM_SEQ));
DEFINE_UNFOLD_SWITCH_FUNC(std::shared_ptr<OpKernelState>, CreateUnfoldOpKernelState);
DEFINE_UNFOLD_SWITCH_FUNC(const void*, GetUnfoldParams);
#undef DEFINE_UNFOLD_SWITCH_FUNC
#undef SWITCH_ENTRY

template<DeviceType device_type, typename T>
class UnfoldKernel final : public OpKernel {
 public:
  UnfoldKernel() = default;
  ~UnfoldKernel() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    const TensorDesc* input_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    if (input_desc->is_dynamic()) { return std::shared_ptr<OpKernelState>(nullptr); }
    const auto& data_format = ctx->Attr<std::string>("data_format");
    const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const auto& stride = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    int spatial_ndim = input_desc->shape().NumAxes() - 2;
    int spatial_dim = GetSpatialDim(ctx->Attr<std::string>("data_format"));
    DataType index_dtype = input_desc->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
                               ? DataType::kInt32
                               : DataType::kInt64;
    return SwitchCreateUnfoldOpKernelState(SwitchCase(index_dtype, spatial_ndim, spatial_dim),
                                           ShapeView(input_desc->shape()), kernel_size,
                                           padding_before, padding_after, stride, dilation);
  }

 private:
  int GetSpatialDim(const std::string& data_format) const {
    int spatial_dim = 0;
    if (data_format == "channels_first") {
      spatial_dim = 2;
    } else if (data_format == "channels_last") {
      spatial_dim = 1;
    } else {
      UNIMPLEMENTED();
    }
    return spatial_dim;
  }

#define SWITCH_ENTRY(func_name, itype, ndim, sdim) \
  UnfoldKernelUtil<device_type, T, itype, ndim, sdim>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, Forward, SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ),
                            MAKE_NDIM_CTRV_SEQ(SPATIAL_NDIM_SEQ),
                            MAKE_NDIM_CTRV_SEQ(SPATIAL_DIM_SEQ));
#undef SWITCH_ENTRY

  void Compute(KernelComputeContext* ctx, OpKernelState* state) const override {
    const Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);
    int spatial_ndim = input->shape().NumAxes() - 2;
    int spatial_dim = GetSpatialDim(ctx->Attr<std::string>("data_format"));
    DataType index_dtype = input->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
                               ? DataType::kInt32
                               : DataType::kInt64;
    auto switch_case = SwitchCase(index_dtype, spatial_ndim, spatial_dim);
    std::shared_ptr<OpKernelState> state_ptr(nullptr);
    if (state == nullptr) {
      const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
      const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
      const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
      const auto& stride = ctx->Attr<std::vector<int32_t>>("strides");
      const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
      state_ptr = SwitchCreateUnfoldOpKernelState(switch_case, input->shape(), kernel_size,
                                                  padding_before, padding_after, stride, dilation);
      state = state_ptr.get();
    }
    const void* params = SwitchGetUnfoldParams(switch_case, state);
    SwitchForward(switch_case, ctx->device_ctx(), params, input->dptr<T>(), output->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class UnfoldGradKernel final : public user_op::OpKernel {
 public:
  UnfoldGradKernel() = default;
  ~UnfoldGradKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    UNIMPLEMENTED();
    return nullptr;
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UNIMPLEMENTED();
  };
};

}  // namespace

#define REGISTER_UNFOLD_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("unfold").SetCreateFn<UnfoldKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                    \
      & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));                        \
  REGISTER_USER_KERNEL("unfold_grad")                                                        \
      .SetCreateFn<UnfoldGradKernel<device, dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                   \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, float)
REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, float)
REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, double)
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
