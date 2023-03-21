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

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class MluNormalizationKernel final : public user_op::OpKernel {
 public:
  MluNormalizationKernel() = default;
  ~MluNormalizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    // make sure input tensor's format NCHW, so channel axis must be 1
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ(axis, 1);
    const auto epsilon = ctx->Attr<float>("epsilon");

    int n = 0, c = 0, h = 0, w = 0;
    if (x->shape_view().NumAxes() == 2) {
      n = x->shape_view().At(0);
      h = 1;
      w = 1;
      c = x->shape_view().At(1);
    } else {
      n = x->shape_view().At(0);
      c = x->shape_view().At(1);
      h = x->shape_view().At(2);
      w = x->shape_view().At(3);
    }

    size_t tmp_in_size = x->shape_view().elem_cnt() * GetSizeOfDataType(x->data_type());
    size_t tmp_out_size = y->shape_view().elem_cnt() * GetSizeOfDataType(y->data_type());
    CnnlWorkspace tmp_in_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_size);
    CnnlWorkspace tmp_out_workspace(ctx->stream()->As<ep::MluStream>(), tmp_out_size);
    void* tmp_in_dptr = tmp_in_workspace.dptr();
    void* tmp_out_dptr = tmp_out_workspace.dptr();

    auto transpose = NewPermutePrimitive(ctx, x->shape_view().NumAxes());
    CHECK(transpose);

    int permute_nhwc[4] = {0, 2, 3, 1};
    // transpose input NCHW -> NHWC
    transpose->Launch(ctx->stream(), x->data_type(), x->shape_view().NumAxes(),
                      x->shape_view().data(), x->dptr<T>(), permute_nhwc, tmp_in_dptr);

    int64_t shape_nhwc[4] = {n, h, w, c};
    CnnlTensorDescriptor input_desc, output_desc, weight_bias_mean_var_desc;
    auto dtype = ConvertToCnnlDataType(x->data_type());
    input_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    output_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    int dim[1] = {c};
    weight_bias_mean_var_desc.set(1, dim, dtype, CNNL_LAYOUT_ARRAY);
    // inference
    OF_CNNL_CHECK(cnnlBatchNormForwardInference(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, input_desc.desc(),
        tmp_in_dptr, weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(),
        moving_mean->dptr(), moving_variance->dptr(), epsilon, output_desc.desc(), tmp_out_dptr));

    int permute_nchw[4] = {0, 3, 1, 2};
    // transpose output NHWC -> NCHW
    transpose->Launch(ctx->stream(), y->data_type(), y->shape_view().NumAxes(), shape_nhwc,
                      tmp_out_dptr, permute_nchw, y->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_IINFERENCE_MLU_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("normalization")                               \
      .SetCreateFn<MluNormalizationKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_BN_IINFERENCE_MLU_KERNEL(float)
REGISTER_BN_IINFERENCE_MLU_KERNEL(float16)

#undef REGISTER_BN_IINFERENCE_MLU_KERNEL

}  // namespace oneflow
