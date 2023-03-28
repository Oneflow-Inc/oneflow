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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

template<typename T>
class MluMultiReduceSumPowAbsKernel final : public user_op::OpKernel {
 public:
  MluMultiReduceSumPowAbsKernel() = default;
  ~MluMultiReduceSumPowAbsKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    size_t input_num = ctx->input_size("x");
    std::vector<const T*> inputs(input_num);
    std::vector<int64_t> sizes(input_num);
    for (size_t i = 0; i < input_num; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      inputs[i] = x->dptr<T>();
      sizes[i] = x->shape_view().elem_cnt();
    }
    float p = ctx->Attr<float>("p");
    auto* stream = ctx->stream()->As<ep::MluStream>();
    size_t workspace_size = input_num * sizeof(T);
    CnnlWorkspace workspace(stream, workspace_size);

    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());
    bang_multi_reduce_sum_pow_abs_kernel<T>(handle, input_num, inputs.data(), sizes.data(),
                                            y->mut_dptr<T>(), p, workspace.dptr(), workspace_size);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("multi_reduce_sum_pow_abs")                    \
      .SetCreateFn<MluMultiReduceSumPowAbsKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL(float)

#undef REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL

}  // namespace
}  // namespace oneflow
