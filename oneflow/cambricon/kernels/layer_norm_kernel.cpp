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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class LayerNormMluKernel final : public user_op::OpKernel {
 public:
  LayerNormMluKernel() = default;
  ~LayerNormMluKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
    const DataType data_type = in->data_type();
    CnnlTensorDescriptor input_desc, output_desc, mean_rstd_desc, filter_bias_desc;
    input_desc.set(in, ConvertToCnnlDataType(data_type));
    output_desc.set(out, ConvertToCnnlDataType(data_type));
    mean_rstd_desc.set(mean, ConvertToCnnlDataType(data_type));

    const void* gamma_dptr = nullptr;
    const void* beta_dptr = nullptr;

    const double eps = ctx->Attr<double>("epsilon");
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = in->shape_view().elem_cnt() / num_instances;
    if (ctx->has_input("gamma", 0)) {
      auto gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      filter_bias_desc.set(gamma, ConvertToCnnlDataType(data_type));
      gamma_dptr = gamma->dptr();

      CHECK_EQ(gamma->shape_view().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) {
      auto beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
      filter_bias_desc.set(beta, ConvertToCnnlDataType(data_type));
      beta_dptr = beta->dptr();
    }

    size_t tmp_buffer_size = 0;
    OF_CNNL_CHECK(cnnlGetLayerNormOpWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                                  begin_norm_axis, input_desc.desc(),
                                                  &tmp_buffer_size));
    CnnlWorkspace cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_buffer_size);

    OF_CNNL_CHECK(cnnlLayerNormForward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(), in->dptr(),
        begin_norm_axis, filter_bias_desc.desc(), gamma_dptr, beta_dptr, eps, cnnl_workspace.dptr(),
        tmp_buffer_size, output_desc.desc(), out->mut_dptr(), mean_rstd_desc.desc(),
        mean->mut_dptr(), inv_variance->mut_dptr()));
  };
};
enum class LayerNormGradRelatedKernelType {
  kDefault,
  kParams,
};
template<LayerNormGradRelatedKernelType TYPE, typename T>
class MluLayerNormGradRelatedKernel final : public user_op::OpKernel {
 public:
  MluLayerNormGradRelatedKernel() = default;
  ~MluLayerNormGradRelatedKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);

    const void* gamma_dptr = nullptr;
    void* dx_mut_dptr = nullptr;
    void* gamma_diff_mut_dptr = nullptr;
    void* beta_diff_mut_dptr = nullptr;

    int64_t axis = 0;
    if constexpr (TYPE == LayerNormGradRelatedKernelType::kDefault) {
      axis = ctx->Attr<int64_t>("begin_norm_axis");
    } else {
      axis = ctx->Attr<int64_t>("begin_params_axis");
    }

    CnnlTensorDescriptor x_desc(x), dy_desc(dy), gamma_desc,
        mean_desc(mean, ConvertToCnnlDataType(x->data_type())), dx_desc;

    const auto stream = ctx->stream()->As<ep::MluStream>();
    CnnlWorkspace gamma_worksacpe(stream),
        dx_workspace(stream, dy->shape_view().Count(0) * GetSizeOfDataType(dy->data_type()));

    if (ctx->has_input("gamma", 0)) {
      auto gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_desc.set(gamma);
      gamma_dptr = gamma->dptr();
    } else {
      const std::vector<int> gamma_shape(x->shape_view().begin() + axis, x->shape_view().end());
      gamma_desc.set(gamma_shape.size(), gamma_shape.data(), ConvertToCnnlDataType(x->data_type()));
      gamma_worksacpe.resize(x->shape_view().Count(axis) * GetSizeOfDataType(x->data_type()));
      gamma_dptr = gamma_worksacpe.dptr();
    }

    if (ctx->has_output("dx", 0)) {
      auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
      dx_desc.set(dx);
      dx_mut_dptr = dx->mut_dptr();
    } else {
      dx_desc.set(dy);
      dx_mut_dptr = dx_workspace.dptr();
    }

    if (ctx->has_output("gamma_diff", 0)) {
      gamma_diff_mut_dptr = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0)->mut_dptr();
    } else {
      gamma_diff_mut_dptr = dx_workspace.dptr();
    }

    if (ctx->has_output("beta_diff", 0)) {
      beta_diff_mut_dptr = ctx->Tensor4ArgNameAndIndex("beta_diff", 0)->mut_dptr();
    } else {
      beta_diff_mut_dptr = dx_workspace.dptr();
    }

    const auto cnnl_handle = stream->cnnl_handle();

    size_t workspace_size = 0;
    OF_CNNL_CHECK(
        cnnlGetLayerNormBackwardWorkspaceSize(cnnl_handle, x_desc.desc(), axis, &workspace_size));
    CnnlWorkspace cnnl_workspace(stream, workspace_size);

    OF_CNNL_CHECK(cnnlLayerNormBackward_v2(
        cnnl_handle, x_desc.desc(), x->dptr(), axis, dy_desc.desc(), dy->dptr(), gamma_desc.desc(),
        gamma_dptr, mean_desc.desc(), mean->dptr(), inv_variance->dptr(), cnnl_workspace.dptr(),
        workspace_size, dx_desc.desc(), dx_mut_dptr, gamma_diff_mut_dptr, beta_diff_mut_dptr));

    if (dx_mut_dptr && ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      auto bcast_add =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kAdd, x->data_type(),
              add_to_output->data_type(), x->shape_view().NumAxes());
      CHECK(bcast_add);
      bcast_add->Launch(ctx->stream(), x->shape_view().NumAxes(), x->shape_view().ptr(),
                        dx_mut_dptr, add_to_output->shape_view().NumAxes(),
                        add_to_output->shape_view().ptr(), add_to_output->dptr(), dx_mut_dptr);
    }
  }
};

#define REGISTER_MLU_LAYER_NORM_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                  \
      .SetCreateFn<LayerNormMluKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MLU_LAYER_NORM_KERNEL(float)
REGISTER_MLU_LAYER_NORM_KERNEL(float16)

#define REGISTER_MLU_LAYER_NORM_GRAD_RELATED_KERNEL(name, type, dtype) \
  REGISTER_USER_KERNEL(name)                                           \
      .SetCreateFn<MluLayerNormGradRelatedKernel<type, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MLU_LAYER_NORM_GRAD_RELATED_KERNEL("layer_norm_grad",
                                            LayerNormGradRelatedKernelType::kDefault, float)
REGISTER_MLU_LAYER_NORM_GRAD_RELATED_KERNEL("layer_norm_grad",
                                            LayerNormGradRelatedKernelType::kDefault, float16)

REGISTER_MLU_LAYER_NORM_GRAD_RELATED_KERNEL("layer_norm_param_grad",
                                            LayerNormGradRelatedKernelType::kParams, float)
REGISTER_MLU_LAYER_NORM_GRAD_RELATED_KERNEL("layer_norm_param_grad",
                                            LayerNormGradRelatedKernelType::kParams, float16)
}  // namespace oneflow
