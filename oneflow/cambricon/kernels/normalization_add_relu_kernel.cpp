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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

enum class BatchNormType {
  kTraining,
  kInference,
};

template<typename T, BatchNormType Type>
class MluNormalizationAddReluKernel final : public user_op::OpKernel {
 public:
  MluNormalizationAddReluKernel() = default;
  ~MluNormalizationAddReluKernel() = default;

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
    const DataType param_data_type = gamma->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_EQ(beta->data_type(), param_data_type);
    CHECK_LT(axis, x->shape_view().NumAxes());
    // axis is equal to 1 for NCHW and equal to 3 for NHWC
    CHECK(axis == 1 || axis == 3);
    int ndim = x->shape_view().size();

    const user_op::Tensor* addend = nullptr;
    if (ctx->has_input("addend", 0)) {
      addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
      CHECK_EQ(x->shape_view(), addend->shape_view());
      CHECK_EQ(addend->data_type(), data_type);
    }

    void* moving_mean_ptr = nullptr;
    void* moving_variance_ptr = nullptr;
    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
      moving_mean_ptr = moving_mean->mut_dptr();
      moving_variance_ptr = moving_variance->mut_dptr();
    }
    auto shape = Shape(x->shape_view());
    const void* x_ptr = x->dptr();
    const void* addend_ptr = addend ? addend->dptr() : nullptr;
    void* y_ptr = y->mut_dptr();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);

    CnnlTensorDescriptor x_desc, y_desc, addend_desc, weight_bias_mean_var_desc;
    const auto stream = ctx->stream()->As<ep::MluStream>();
    CnnlWorkspace workspace_x(stream, 0), workspace_y(stream, 0), workspace_addend(stream, 0);
    if (axis == 1) {
      shape = mlu::ComputeShapeNchwToNhwc(shape);
      size_t workspace_size = shape.elem_cnt() * GetSizeOfDataType(data_type);
      workspace_x.resize(workspace_size);
      workspace_y.resize(workspace_size);
      // convert x to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), x->shape_view(), data_type, x->dptr(),
                               workspace_x.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      x_ptr = workspace_x.dptr();
      y_ptr = workspace_y.dptr();
      if (addend) {
        workspace_addend.resize(workspace_size);
        mlu::ConvertMemoryFormat(ctx->stream(), addend->shape_view(), data_type, addend->dptr(),
                                 workspace_addend.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
        addend_ptr = workspace_addend.dptr();
      }
    }
    x_desc.set(ndim, shape.data(), cnnl_data_type, CNNL_LAYOUT_NHWC);
    y_desc.set(ndim, shape.data(), cnnl_data_type, CNNL_LAYOUT_NHWC);
    if (addend) { addend_desc.set(ndim, shape.data(), cnnl_data_type, CNNL_LAYOUT_NHWC); }
    int64_t dims[1] = {shape[ndim - 1]};
    weight_bias_mean_var_desc.set(1, dims, ConvertToCnnlDataType(gamma->data_type()),
                                  CNNL_LAYOUT_ARRAY);
    CnnlActivationDescriptor activation_desc;
    activation_desc.set(CNNL_ACTIVATION_RELU, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                        /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);

    size_t bn_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetBatchNormForwardWorkspaceSize(stream->cnnl_handle(), x_desc.desc(),
                                                       &bn_workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), bn_workspace_size);

    auto op_mode = addend ? CNNL_BATCHNORM_OPS_BN_ADD_ACTIVATION : CNNL_BATCHNORM_OPS_BN_ACTIVATION;
    if constexpr (Type == BatchNormType::kTraining) {
      CHECK(training);
      auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
      auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
      CHECK_EQ(mean->data_type(), param_data_type);
      CHECK_EQ(inv_variance->data_type(), param_data_type);
      const auto momentum = ctx->Attr<float>("momentum");
      OF_CNNL_CHECK(cnnlBatchNormForwardTrainingV2(
          stream->cnnl_handle(), activation_desc.desc(), CNNL_BATCHNORM_SPATIAL, op_mode, nullptr,
          nullptr, x_desc.desc(), x_ptr, addend_desc.desc(), addend_ptr,
          weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(), moving_mean_ptr,
          moving_variance_ptr, epsilon, momentum, y_desc.desc(), y_ptr, mean->mut_dptr(),
          inv_variance->mut_dptr(), workspace.dptr(), bn_workspace_size, nullptr, 0));
    } else {
      CHECK(!training);
      OF_CNNL_CHECK(cnnlBatchNormForwardInferenceV2(
          stream->cnnl_handle(), activation_desc.desc(), CNNL_BATCHNORM_SPATIAL, op_mode, nullptr,
          nullptr, x_desc.desc(), x_ptr, weight_bias_mean_var_desc.desc(), gamma->dptr(),
          beta->dptr(), addend_desc.desc(), addend_ptr, moving_mean_ptr, moving_variance_ptr,
          epsilon, y_desc.desc(), y_ptr));
    }
    if (axis == 1) {
      // convert y to NCHW
      mlu::ConvertMemoryFormat(ctx->stream(), shape, data_type, y_ptr, y->mut_dptr(),
                               MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_RELU_MLU_KERNEL(dtype, type, training)                            \
  REGISTER_USER_KERNEL("normalization_relu")                                          \
      .SetCreateFn<MluNormalizationAddReluKernel<dtype, type>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<bool>("training") == training));

REGISTER_BN_RELU_MLU_KERNEL(float, BatchNormType::kTraining, true)
REGISTER_BN_RELU_MLU_KERNEL(float16, BatchNormType::kTraining, true)
REGISTER_BN_RELU_MLU_KERNEL(float, BatchNormType::kInference, false)
REGISTER_BN_RELU_MLU_KERNEL(float16, BatchNormType::kInference, false)

#undef REGISTER_BN_RELU_MLU_KERNEL

#define REGISTER_BN_ADD_RELU_MLU_KERNEL(dtype, type, training)                        \
  REGISTER_USER_KERNEL("normalization_add_relu")                                      \
      .SetCreateFn<MluNormalizationAddReluKernel<dtype, type>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<bool>("training") == training));

REGISTER_BN_ADD_RELU_MLU_KERNEL(float, BatchNormType::kTraining, true)
REGISTER_BN_ADD_RELU_MLU_KERNEL(float16, BatchNormType::kTraining, true)
REGISTER_BN_ADD_RELU_MLU_KERNEL(float, BatchNormType::kInference, false)
REGISTER_BN_ADD_RELU_MLU_KERNEL(float16, BatchNormType::kInference, false)

#undef REGISTER_BN_ADD_RELU_MLU_KERNEL

}  // namespace oneflow
