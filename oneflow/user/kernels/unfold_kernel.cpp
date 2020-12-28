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

class UnfoldOpKernelState final : public user_op::OpKernelState {
 public:
  UnfoldOpKernelState() {}

  template<size_t NDims>
  static std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(
      user_op::KernelInitContext* ctx) {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    std::vector<int32_t> padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int32_t idx_offset = IdxOffset(data_format);
    const size_t c_dim = data_format == "channels_first" ? 1 : NDims + 1;
    std::shared_ptr<UnfoldOpKernelState> state(new UnfoldOpKernelState());

    auto Gen5DShape = [](const Shape& shape, int32_t idx_offset) -> Shape {
      DimVector ret_vec(shape.dim_vec());
      int32_t ndims = ret_vec.size() - 2;
      ret_vec.insert(ret_vec.begin() + idx_offset, 3 - ndims, 1);
      return Shape(ret_vec);
    };
    auto Gen3DVec = [](const std::vector<int32_t>& origin_vec,
                       int32_t num) -> std::vector<int32_t> {
      std::vector<int32_t> ret_vec = origin_vec;
      ret_vec.insert(ret_vec.begin(), 3 - ret_vec.size(), num);
      return ret_vec;
    };

    DimVector native_shape(NDims + 2);
    native_shape.at(0) = x_shape.At(0);
    native_shape.at(c_dim) = x_shape.At(c_dim);
    for (int32_t i = 0; i < NDims; ++i) {
      native_shape[idx_offset + i] =
          (x_shape.At(idx_offset + i) + padding_before[i] + padding_after[i]
           - dilation_rate[i] * (kernel_size[i] - 1) - 1)
              / strides[i]
          + 1;
    }

    state->in_5d_shape_ = Gen5DShape(x_shape, idx_offset);
    state->out_5d_shape_ = Gen5DShape(Shape(native_shape), idx_offset);
    state->out_shape_ = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
    state->kernel_size_3d_ = Gen3DVec(kernel_size, 1);
    state->strides_3d_ = Gen3DVec(strides, 1);
    state->dilation_rate_3d_ = Gen3DVec(dilation_rate, 1);
    state->padding_before_3d_ = Gen3DVec(padding_before, 0);

    return std::move(state);
  }

 public:
  Shape in_5d_shape_;
  Shape out_5d_shape_;
  Shape out_shape_;
  std::vector<int32_t> kernel_size_3d_;
  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> dilation_rate_3d_;
  std::vector<int32_t> padding_before_3d_;
};

template<DeviceType device_type, typename T>
class UnfoldKernelImpl {
 public:
  static void BWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* unfold_state = dynamic_cast<UnfoldOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    if (data_format == "channels_first") {
      UnfoldKernelUtil<device_type, T>::CFirstBackward(
          ctx->device_ctx(), unfold_state->in_5d_shape_, unfold_state->out_5d_shape_,
          unfold_state->out_shape_, unfold_state->kernel_size_3d_, unfold_state->strides_3d_,
          unfold_state->dilation_rate_3d_, unfold_state->padding_before_3d_, dy->dptr<T>(),
          dx->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }
};

template<size_t NDims, DeviceType device_type, typename T>
class UnfoldGradKernel final : public user_op::OpKernel {
 public:
  UnfoldGradKernel() = default;
  ~UnfoldGradKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldOpKernelState::DoCreateOpKernelState<NDims>(ctx);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldKernelImpl<device_type, T>::BWCompute(ctx, state);
  };
};

// new implementation

template<typename INDEX_T, int NDIM, int SDIM>
class UnfoldOpKernelStateV2 : public OpKernelState {
 public:
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  UnfoldOpKernelStateV2(const ShapeView& input_shape, const std::vector<int32_t>& kernel_size,
                        const std::vector<int32_t>& padding_before,
                        const std::vector<int32_t>& padding_after,
                        const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation)
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
  return std::make_shared<UnfoldOpKernelStateV2<INDEX_T, NDIM, SDIM>>(
      input_shape, kernel_size, padding_before, padding_after, stride, dilation);
}

template<typename INDEX_T, int NDIM, int SDIM>
const void* GetUnfoldParams(OpKernelState* state) {
  auto* unfold_state = dynamic_cast<UnfoldOpKernelStateV2<INDEX_T, NDIM, SDIM>*>(state);
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
class UnfoldKernelV2 final : public OpKernel {
 public:
  UnfoldKernelV2() = default;
  ~UnfoldKernelV2() = default;

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
  UnfoldKernelUtilV2<device_type, T, itype, ndim, sdim>::func_name
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

}  // namespace

#define REGISTER_UNFOLD_KERNEL_V2(device, dtype)                                               \
  REGISTER_USER_KERNEL("unfold").SetCreateFn<UnfoldKernelV2<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                      \
      & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_KERNEL_V2(DeviceType::kCPU, float)
REGISTER_UNFOLD_KERNEL_V2(DeviceType::kCPU, double)

REGISTER_UNFOLD_KERNEL_V2(DeviceType::kGPU, float)
REGISTER_UNFOLD_KERNEL_V2(DeviceType::kGPU, double)

#define REGISTER_UNFOLD_KERNEL_NDIMS(dim, device, dtype)   \
  REGISTER_USER_KERNEL("unfold_grad")                      \
      .SetCreateFn<UnfoldGradKernel<dim, device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kCPU, float)
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kGPU, float)
REGISTER_UNFOLD_KERNEL_NDIMS(2, DeviceType::kGPU, double)
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
