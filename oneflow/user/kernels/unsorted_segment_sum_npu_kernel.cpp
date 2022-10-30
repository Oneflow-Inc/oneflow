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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T, typename K>
class UnsortedSegmentSumNpuKernel final : public user_op::OpKernel {
 public:
  UnsortedSegmentSumNpuKernel() = default;
  ~UnsortedSegmentSumNpuKernel() override = default;
 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    int64_t axis = ctx->Attr<int64_t>("axis");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t outer_dim_size = out->shape_view().Count(0, axis);
    int64_t num_segments = out->shape_view().At(axis);
    int64_t inner_dim_size = out->shape_view().Count(axis + 1);
    int64_t num_segment_ids = segment_ids->shape_view().elem_cnt();
    NpuCommand npu_command;
    npu_command.OpName("EmbeddingDenseGrad")
               .Input(data)
               .Input(segment_ids)
               .Attr("num_weights", static_cast<int64_t>(like->shape_view().At(0)))
               .Attr("padding_idx", static_cast<int64_t>(-1))
               .Attr("scale_grad_by_freq", false)
               .Output(out)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_NPU_KERNEL(out_type, segment_ids_type, kernel_type) \
  REGISTER_USER_KERNEL(kernel_type)                                                           \
      .SetCreateFn<UnsortedSegmentSumNpuKernel<DeviceType::kNPU, out_type,               \
                                            segment_ids_type>>()            \
      .SetIsMatchedHob(                                                                       \
          (user_op::HobDeviceType() == DeviceType::kNPU)                                                \
          && (user_op::HobDataType("segment_ids", 0) == GetDataType<segment_ids_type>::value)  \
          && (user_op::HobDataType("out", 0) == GetDataType<out_type>::value));


REGISTER_UNSORTED_SEGMENT_SUM_NPU_KERNEL(float, int, "unsorted_segment_sum_like")
REGISTER_UNSORTED_SEGMENT_SUM_NPU_KERNEL(float16, int, "unsorted_segment_sum_like")
}

}