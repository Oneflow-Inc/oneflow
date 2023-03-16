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
    mean_rstd_desc.set(inv_variance, ConvertToCnnlDataType(data_type));

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

#define REGISTER_LAYER_NORM_MLU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                  \
      .SetCreateFn<LayerNormMluKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_MLU_KERNEL(float)
REGISTER_LAYER_NORM_MLU_KERNEL(float16)

}  // namespace oneflow
