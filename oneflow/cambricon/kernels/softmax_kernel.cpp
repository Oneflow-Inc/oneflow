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
#include "cnnl.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluSoftmaxKernel final : public user_op::OpKernel {
 public:
  MluSoftmaxKernel() = default;
  ~MluSoftmaxKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CnnlTensorDescriptor input_desc, output_desc;
    int in_ndim = in->shape_view().NumAxes();
    int batch_dims = in->shape_view().Count(0, in_ndim - 1);
    int softmax_dim = in->shape_view().At(in_ndim - 1);

    std::vector<int> addentional_dims = {batch_dims, 1, softmax_dim};
    input_desc.set_reshape(in, addentional_dims);
    output_desc.set_reshape(out, addentional_dims);

    OF_CNNL_CHECK(cnnlSoftmaxForward_v2(
        /* handle    */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* algorithm */ CNNL_SOFTMAX_ACCURATE,
        /* mode      */ CNNL_SOFTMAX_MODE_LOW_DIMENSION,
        /* prefer    */ CNNL_COMPUTATION_HIGH_PRECISION,
        /* alpha     */ nullptr,
        /* x_desc    */ input_desc.desc(),
        /* x         */ in->dptr(),
        /* beta      */ NULL,
        /* y_desc    */ output_desc.desc(),
        /* y         */ out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_MLU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("softmax").SetCreateFn<MluSoftmaxKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                      \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_SOFTMAX_MLU_KERNEL(float)
REGISTER_SOFTMAX_MLU_KERNEL(float16)

template<typename T>
class MluSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  MluSoftmaxGradKernel() = default;
  ~MluSoftmaxGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int y_ndim = y->shape_view().NumAxes();
    int batch_dims = y->shape_view().Count(0, y_ndim - 1);
    int softmax_dim = y->shape_view().At(y_ndim - 1);

    CnnlTensorDescriptor y_desc, dy_desc, dx_desc;
    std::vector<int> addentional_dims = {batch_dims, 1, softmax_dim};
    y_desc.set_reshape(y, addentional_dims);
    dy_desc.set_reshape(dy, addentional_dims);
    dx_desc.set_reshape(dx, addentional_dims);

    cnnlSoftmaxBackward(
        /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* algorithm   */ CNNL_SOFTMAX_ACCURATE,
        /* mode        */ CNNL_SOFTMAX_MODE_LOW_DIMENSION,
        /* alpha       */ nullptr,
        /* y_desc      */ y_desc.desc(),
        /* y           */ y->dptr(),
        /* diff_y_desc */ dy_desc.desc(),
        /* diff_y      */ dy->dptr(),
        /* beta        */ nullptr,
        /* diff_x_desc */ dx_desc.desc(),
        /* diff_x      */ dx->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GRAD_MLU_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("softmax_grad")                                \
      .SetCreateFn<MluSoftmaxGradKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_SOFTMAX_GRAD_MLU_KERNEL(float)
REGISTER_SOFTMAX_GRAD_MLU_KERNEL(float16)
}  // namespace oneflow
