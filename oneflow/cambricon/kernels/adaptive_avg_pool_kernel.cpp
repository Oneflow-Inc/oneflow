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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class AdaptiveAvgPool2DKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DKernel() = default;
  ~AdaptiveAvgPool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    size_t tmp_in_workspace_size =
        in_tensor->shape_view().elem_cnt() * sizeof(in_tensor->data_type());
    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_workspace_size);
    void* tmp_in_ptr = tmp_in_cnnl_workspace.dptr();

    std::vector<int64_t> in_shapevec({in_tensor->shape_view().At(0), in_tensor->shape_view().At(1),
                                      in_tensor->shape_view().At(2),
                                      in_tensor->shape_view().At(3)});
    auto transpose = NewPermutePrimitive(ctx, in_tensor->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in_tensor->data_type(), in_tensor->shape_view().NumAxes(),
                      in_shapevec.data(), in_ptr, std::vector<int>({0, 2, 3, 1}).data(),
                      tmp_in_ptr);
    cnnlTensorDescriptor_t in_desc = nullptr, out_decs = nullptr;
    const int in_dims[4] = {static_cast<int>(in_tensor->shape_view().At(0)),
                            static_cast<int>(in_tensor->shape_view().At(2)),
                            static_cast<int>(in_tensor->shape_view().At(3)),
                            static_cast<int>(in_tensor->shape_view().At(1))};
    const int out_dims[4] = {static_cast<int>(out_tensor->shape_view().At(0)),
                             static_cast<int>(out_tensor->shape_view().At(2)),
                             static_cast<int>(out_tensor->shape_view().At(3)),
                             static_cast<int>(out_tensor->shape_view().At(1))};
    auto dtype = ConvertToCnnlDataType(in_tensor->data_type());
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&in_desc));
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&out_decs));
    OF_CNNL_CHECK(cnnlSetTensorDescriptor(in_desc, layout, dtype, 4, in_dims));
    OF_CNNL_CHECK(cnnlSetTensorDescriptor(out_decs, layout, dtype, 4, out_dims));
    size_t _adaptive_avg_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc,
        CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, out_decs,
        &_adaptive_avg_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                            _adaptive_avg_pool2d_workspace_size);
    void* _adaptive_avg_pool2d_workspace = adaptive2d_cnnl_workspace.dptr();
    size_t tmp_out_workspace_size =
        out_tensor->shape_view().elem_cnt() * sizeof(in_tensor->data_type());
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                         tmp_out_workspace_size);
    void* tmp_out_ptr = tmp_out_cnnl_workspace.dptr();
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc, tmp_in_ptr,
        CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, _adaptive_avg_pool2d_workspace,
        _adaptive_avg_pool2d_workspace_size, out_decs, tmp_out_ptr, NULL, NULL));
    std::vector<int64_t> out_shapevec(
        {out_tensor->shape_view().At(0), out_tensor->shape_view().At(2),
         out_tensor->shape_view().At(3), out_tensor->shape_view().At(1)});
    transpose = NewPermutePrimitive(ctx, out_tensor->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), out_tensor->data_type(), out_tensor->shape_view().NumAxes(),
                      out_shapevec.data(), tmp_out_ptr, std::vector<int>({0, 3, 1, 2}).data(),
                      out_ptr);
    cnnlDestroyTensorDescriptor(in_desc);
    cnnlDestroyTensorDescriptor(out_decs);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                         \
      .SetCreateFn<AdaptiveAvgPool2DKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float16)

}  // namespace oneflow
