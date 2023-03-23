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
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<typename T>
class MluUnsortedSegmentSumKernel final : public user_op::OpKernel {
 public:
  MluUnsortedSegmentSumKernel() = default;
  ~MluUnsortedSegmentSumKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    const user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    int64_t axis = ctx->Attr<int64_t>("axis");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memset<DeviceType::kMLU>(ctx->stream(), out->mut_dptr(), 0,
                             out->shape_view().elem_cnt() * sizeof(T));

    if (axis != 0) { LOG(FATAL) << "only support axis == 0 for MLU device."; }
    CnnlTensorDescriptor data_desc(data), indices_desc(segment_ids), out_desc(out);
    size_t workspace_size = 0;
    OF_CNNL_CHECK(
        cnnlGetUnsortedSegmentSumWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                               data_desc.desc(), out_desc.desc(), &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlUnsortedSegmentSum(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                         data_desc.desc(), data->dptr(), indices_desc.desc(),
                                         segment_ids->dptr<int32_t>(), workspace.dptr(),
                                         workspace_size, out_desc.desc(), out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(out_type)                                     \
  REGISTER_USER_KERNEL("unsorted_segment_sum_like")                                        \
      .SetCreateFn<MluUnsortedSegmentSumKernel<OF_PP_PAIR_FIRST(out_type)>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                      \
                       && (user_op::HobDataType("segment_ids", 0) == DataType::kInt32)     \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_type))  \
                       && (user_op::HobDataType("data", 0) == OF_PP_PAIR_SECOND(out_type)) \
                       && (user_op::HobDataType("like", 0) == OF_PP_PAIR_SECOND(out_type)));

REGISTER_UNSORTED_SEGMENT_SUM_KERNEL((float, DataType::kFloat))
REGISTER_UNSORTED_SEGMENT_SUM_KERNEL((float16, DataType::kFloat16))

}  // namespace oneflow