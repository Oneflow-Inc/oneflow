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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

// return nhwc shape
template<typename T>
std::vector<int64_t> HandleNCHWTensor(ep::Stream* stream, const user_op::Tensor* t,
                                      CnnlWorkspace& workspace,
                                      const std::unique_ptr<ep::primitive::Permute>& do_transpose) {
  const auto shape_view = t->shape_view();
  std::vector<int64_t> shape_vec(shape_view.begin(), shape_view.end());
  std::swap(shape_vec[1], shape_vec[2]);
  std::swap(shape_vec[2], shape_vec[3]);

  workspace.resize(shape_view.Count(0) * GetSizeOfDataType(t->data_type()));
  if (do_transpose) {
    std::array<int, 4> nchw_to_nhwc{0, 2, 3, 1};
    do_transpose->Launch(stream, t->data_type(), shape_view.NumAxes(), shape_view.data(),
                         t->dptr<T>(), nchw_to_nhwc.data(), workspace.dptr());
  }
  return shape_vec;
}

template<typename T>
void HandleNHWCTensor(ep::Stream* stream, user_op::Tensor* t,
                      const std::vector<int64_t>& nhwc_shape, CnnlWorkspace& workspace,
                      const std::unique_ptr<ep::primitive::Permute>& do_transpose) {
  if (do_transpose) {
    std::array<int, 4> nhwc_to_nchw{0, 3, 1, 2};
    do_transpose->Launch(stream, t->data_type(), nhwc_shape.size(), nhwc_shape.data(),
                         workspace.dptr(), nhwc_to_nchw.data(), t->mut_dptr());
  }
}

void AssignTensorDescriptorWithNHWCShape(CnnlTensorDescriptor& desc, DataType dtype,
                                         const std::vector<int64_t>& nhwc_shape) {
  desc.set(4, nhwc_shape.data(), ConvertToCnnlDataType(dtype), CNNL_LAYOUT_NHWC);
}

}  // namespace

enum class BatchNormType {
  kTraining,
  kInference,
};

template<typename T, BatchNormType Type>
class MluNormalizationKernel final : public user_op::OpKernel {
 public:
  MluNormalizationKernel() = default;
  ~MluNormalizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const bool training = ctx->Attr<bool>("training");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_LT(axis, x->shape_view().NumAxes());
    // make sure input tensor's format NCHW, so channel axis must be 1
    CHECK_EQ(axis, 1);

    void* moving_mean_ptr = nullptr;
    void* moving_variance_ptr = nullptr;

    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
      moving_mean_ptr = moving_mean->mut_dptr();
      moving_variance_ptr = moving_variance->mut_dptr();
    }

    const auto stream = ctx->stream()->As<ep::MluStream>();

    const auto transpose = NewPermutePrimitive(ctx, x->shape_view().NumAxes());

    CnnlWorkspace workspace_x(stream, 0), workspace_y(stream, 0);

    const auto new_x_shape = HandleNCHWTensor<T>(stream, x, workspace_x, transpose);
    const auto new_y_shape = HandleNCHWTensor<T>(stream, y, workspace_y, nullptr);

    CnnlTensorDescriptor x_desc, y_desc, weight_bias_mean_var_desc;
    const auto dtype = x->data_type();
    const auto assign_desc = [dtype](CnnlTensorDescriptor& desc,
                                     const std::vector<int64_t>& nhwc_shape) {
      return AssignTensorDescriptorWithNHWCShape(desc, dtype, nhwc_shape);
    };
    assign_desc(x_desc, new_x_shape);
    assign_desc(y_desc, new_y_shape);
    int dim[1] = {new_x_shape[new_x_shape.size() - 1]};
    weight_bias_mean_var_desc.set(1, dim, ConvertToCnnlDataType(dtype), CNNL_LAYOUT_ARRAY);

    if constexpr (Type == BatchNormType::kTraining) {
      CHECK(training);
      auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
      auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
      const auto momentum = ctx->Attr<float>("momentum");
      // api reference:
      // https://www.cambricon.com/docs/sdk_1.10.0/cambricon_cnnl_1.15.2/developer_guide/cnnl_api/api/batchnorm.html#cnnlbatchnormforwardtraining
      // training
      OF_CNNL_CHECK(cnnlBatchNormForwardTraining(
          ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, x_desc.desc(),
          workspace_x.dptr(), weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(),
          moving_mean_ptr, moving_variance_ptr, epsilon, momentum, y_desc.desc(),
          workspace_y.dptr(), mean->mut_dptr(), inv_variance->mut_dptr()));
    } else {
      CHECK(!training);
      // api reference:
      // https://www.cambricon.com/docs/sdk_1.10.0/cambricon_cnnl_1.15.2/developer_guide/cnnl_api/api/batchnorm.html#cnnlbatchnormforwardinference
      // inference
      OF_CNNL_CHECK(cnnlBatchNormForwardInference(
          ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, x_desc.desc(),
          workspace_x.dptr(), weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(),
          moving_mean_ptr, moving_variance_ptr, epsilon, y_desc.desc(), workspace_y.dptr()));
    }
    HandleNHWCTensor<T>(stream, y, new_y_shape, workspace_y, transpose);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_MLU_KERNEL(dtype, type, training)                                 \
  REGISTER_USER_KERNEL("normalization")                                               \
      .SetCreateFn<MluNormalizationKernel<dtype, type>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<bool>("training") == training));

REGISTER_BN_MLU_KERNEL(float, BatchNormType::kTraining, true)
REGISTER_BN_MLU_KERNEL(float16, BatchNormType::kTraining, true)
REGISTER_BN_MLU_KERNEL(float, BatchNormType::kInference, false)
REGISTER_BN_MLU_KERNEL(float16, BatchNormType::kInference, false)

#undef REGISTER_BN_MLU_KERNEL

template<typename T>
class MluNormalizationGradKernel final : public user_op::OpKernel {
 public:
  MluNormalizationGradKernel() = default;
  ~MluNormalizationGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);

    auto* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    auto* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto epsilon = ctx->Attr<float>("epsilon");

    const auto stream = ctx->stream()->As<ep::MluStream>();

    const auto transpose = NewPermutePrimitive(ctx, x->shape_view().NumAxes());

    CnnlWorkspace workspace_x(stream, 0), workspace_dy(stream, 0), workspace_dx(stream, 0);

    const auto new_x_shape = HandleNCHWTensor<T>(stream, x, workspace_x, transpose);
    const auto new_dy_shape = HandleNCHWTensor<T>(stream, dy, workspace_dy, transpose);
    const auto new_dx_shape = HandleNCHWTensor<T>(stream, dx, workspace_dx, nullptr);

    CnnlTensorDescriptor x_desc, dy_desc, gamma_desc(gamma), dx_desc;
    const auto assign_desc = [dtype = x->data_type()](CnnlTensorDescriptor& desc,
                                                      const std::vector<int64_t>& nhwc_shape) {
      return AssignTensorDescriptorWithNHWCShape(desc, dtype, nhwc_shape);
    };
    assign_desc(x_desc, new_x_shape);
    assign_desc(dy_desc, new_dy_shape);
    assign_desc(dx_desc, new_dx_shape);

    cnnlBatchNormBackward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr,
                          nullptr, nullptr, x_desc.desc(), workspace_x.dptr(), dy_desc.desc(),
                          workspace_dy.dptr(), gamma_desc.desc(), gamma->dptr(), mean->dptr(),
                          inv_variance->dptr(), epsilon, dx_desc.desc(), workspace_dx.dptr(),
                          gamma_diff->mut_dptr(), beta_diff->mut_dptr());
    HandleNHWCTensor<T>(stream, dx, new_dx_shape, workspace_dx, transpose);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("normalization_grad")                          \
      .SetCreateFn<MluNormalizationGradKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_BN_GRAD_KERNEL(float)
REGISTER_BN_GRAD_KERNEL(float16)

#undef REGISTER_BN_GRAD_KERNEL

}  // namespace oneflow
