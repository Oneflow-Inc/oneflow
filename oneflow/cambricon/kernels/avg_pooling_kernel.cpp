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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<int Nd, typename T>
class MluAvgPoolKernel final : public user_op::OpKernel {
 public:
  MluAvgPoolKernel() = default;
  ~MluAvgPoolKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
    const int32_t divisor_override = ctx->Attr<int32_t>("divisor_override");

    CHECK_OR_THROW(padding.size() == 2) << "padding size should be 2.";
    CHECK_OR_THROW(kernel_size.size() == 2) << "kernel_size size should be 2.";
    CHECK_OR_THROW(stride.size() == 2) << "stride size should be 2.";
    CHECK_OR_THROW(divisor_override == 0)
        << "cambricon cnnl avg pool does not support divisor_override.";

    cnnlTensorLayout_t layout =
        (data_format == "channels_last") ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
    cnnlPoolingMode_t mode = count_include_pad ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                               : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    CnnlPoolingDescriptor pooling_desc;
    CnnlTensorDescriptor x_desc, y_desc;
    x_desc.set(x, layout);
    y_desc.set(y, layout);

    int h_axis = (data_format == "channels_last") ? 1 : 2;
    int w_axis = h_axis + 1;
    int64_t output_h = y->shape_view()[h_axis];
    int64_t output_w = y->shape_view()[w_axis];

    // calculate paddings
    int pu = padding[0], pd = padding[0], pl = padding[1], pr = padding[1];
    pooling_desc.set(mode, kernel_size[0], kernel_size[1], stride[0], stride[1], pu, pd, pl, pr,
                     ceil_mode);

    auto handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    size_t pooling_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetPoolingWorkspaceSize(
        /* handle         */ handle,
        /* mode           */ mode,
        /* output_w       */ output_w,
        /* output_h       */ output_h,
        /* workspace_size */ &pooling_workspace_size));
    CnnlWorkspace pooling_workspace(ctx->stream()->As<ep::MluStream>(), pooling_workspace_size);

    const void* extra_device_input_dptr = nullptr;
    CnnlHostWorkspace extra_input_workspace(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace extra_device_input_workspace(ctx->stream()->As<ep::MluStream>());
    size_t extra_input_size = 0;
    OF_CNNL_CHECK(
        cnnlGetPoolingExtraInputSize(handle, mode, output_w, output_h, &extra_input_size));
    if (extra_input_size > 0) {
      extra_input_workspace.resize(extra_input_size);
      OF_CNNL_CHECK(cnnlInitPoolingExtraInput(handle, pooling_desc.desc(), x_desc.desc(),
                                              y_desc.desc(), extra_input_workspace.dptr()));
      extra_device_input_workspace.resize(extra_input_size);
      OF_MLU_CHECK(cnrtMemcpyAsync(
          extra_device_input_workspace.dptr(), extra_input_workspace.dptr(), extra_input_size,
          ctx->stream()->As<ep::MluStream>()->mlu_stream(), cnrtMemcpyHostToDev));
      extra_device_input_dptr = extra_device_input_workspace.dptr();
    }

    OF_CNNL_CHECK(cnnlPoolingForward_v2(
        /* handle         */ handle,
        /* pooling_desc   */ pooling_desc.desc(),
        /* alpha          */ nullptr,
        /* x_desc         */ x_desc.desc(),
        /* x              */ x->dptr(),
        /* beta           */ nullptr,
        /* extra_input    */ extra_device_input_dptr,
        /* y_desc         */ y_desc.desc(),
        /* y              */ y->mut_dptr(),
        /* workspace      */ pooling_workspace.dptr(),
        /* workspace_size */ pooling_workspace_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_AVG_POOL_MLU_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("avg_pool_2d")                                 \
      .SetCreateFn<MluAvgPoolKernel<2, dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_AVG_POOL_MLU_KERNEL(float)
REGISTER_AVG_POOL_MLU_KERNEL(float16)

}  // namespace oneflow
