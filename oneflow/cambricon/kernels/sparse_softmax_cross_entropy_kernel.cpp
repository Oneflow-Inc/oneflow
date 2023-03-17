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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class MluSparseSoftmaxCrossEntropyKernel final : public user_op::OpKernel {
 public:
  MluSparseSoftmaxCrossEntropyKernel() = default;
  ~MluSparseSoftmaxCrossEntropyKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    CHECK_EQ_OR_THROW(label->data_type(), DataType::kInt32)
        << "the data type of label should be int32 for mlu sparse_softmax_cross_entropy op.";

    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_instances = label->shape_view().elem_cnt();
    CHECK_EQ(prediction->shape_view().elem_cnt() % num_instances, 0);

    CnnlTensorDescriptor prediction_desc, label_desc, prob_desc, out_desc;
    prediction_desc.set(prediction);
    label_desc.set(label);
    prob_desc.set(prob);
    out_desc.set(out);

    // prob means the gradients.
    OF_CNNL_CHECK(cnnlSparseSoftmaxCrossEntropyWithLogits_v2(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), CNNL_COMPUTATION_HIGH_PRECISION,
        CNNL_SOFTMAX_MODE_LOW_DIMENSION, prediction_desc.desc(), prediction->dptr(),
        label_desc.desc(), label->dptr(), out_desc.desc(), out->mut_dptr(), prob_desc.desc(),
        prob->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("sparse_softmax_cross_entropy")
    .SetCreateFn<MluSparseSoftmaxCrossEntropyKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)
                     && ((user_op::HobDataType("out", 0) == DataType::kFloat)
                         || (user_op::HobDataType("out", 0) == DataType::kFloat16)));

}  // namespace oneflow
