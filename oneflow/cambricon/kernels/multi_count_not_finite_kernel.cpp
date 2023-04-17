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
class MluMultiCountNotFiniteKernel final : public user_op::OpKernel {
 public:
  MluMultiCountNotFiniteKernel() = default;
  ~MluMultiCountNotFiniteKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    size_t input_num = ctx->inputs().size();
    std::vector<const T*> inputs(input_num);
    std::vector<int64_t> sizes(input_num);
    for (size_t i = 0; i < input_num; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      inputs[i] = x->dptr<T>();
      sizes[i] = x->shape_view().elem_cnt();
    }
    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());
    size_t workspace_size =
        stream->device()->nclusters() * stream->device()->ncores_per_cluster() * sizeof(int64_t);
    CnnlWorkspace workspace(stream, workspace_size);

    if constexpr (std::is_same<T, float16>::value) {
      bang_multi_count_not_finite_half_kernel(
          handle, input_num, reinterpret_cast<const void**>(inputs.data()), sizes.data(),
          y->mut_dptr<int64_t>(), workspace.dptr(), workspace_size);
    } else {
      bang_multi_count_not_finite_kernel<T>(handle, input_num, inputs.data(), sizes.data(),
                                            y->mut_dptr<int64_t>(), workspace.dptr(),
                                            workspace_size);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_MULTI_COUNT_NOT_FINITE_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("multi_count_not_finite")                      \
      .SetCreateFn<MluMultiCountNotFiniteKernel<dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MLU_MULTI_COUNT_NOT_FINITE_KERNEL(float)
REGISTER_MLU_MULTI_COUNT_NOT_FINITE_KERNEL(float16)

#undef REGISTER_MLU_MULTI_COUNT_NOT_FINITE_KERNEL

}  // namespace
}  // namespace oneflow
