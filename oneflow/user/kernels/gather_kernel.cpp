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
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/job/parallel_distribution_util.h"

namespace oneflow {

namespace user_op {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

class GatherOpKernelState final : public user_op::OpKernelState {
 public:
  GatherOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~GatherOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

void CheckParallelDistribution(const Shape& hierarchy, int64_t gather_axis,
                               const ParallelDistribution& in_parallel_distribution,
                               const ParallelDistribution& indices_parallel_distribution,
                               const ParallelDistribution& out_parallel_distribution) {
  CHECK_EQ(hierarchy.NumAxes(), in_parallel_distribution.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), indices_parallel_distribution.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), in_parallel_distribution.sbp_parallel_size());
  if (hierarchy.elem_cnt() == 1) { return; }
  FOR_RANGE(int64_t, i, 0, hierarchy.NumAxes()) {
    const auto& in_sbp = in_parallel_distribution.sbp_parallel(i);
    if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == gather_axis) {
      CHECK(indices_parallel_distribution.sbp_parallel(i).has_broadcast_parallel());
      CHECK(out_parallel_distribution.sbp_parallel(i).has_partial_sum_parallel());
    }
  }
}

}  // namespace

template<DeviceType device_type, typename T, typename K>
class GatherKernel final : public user_op::OpKernel {
 public:
  GatherKernel() = default;
  ~GatherKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto axis = ctx->Attr<int64_t>("axis");
    const ParallelDistribution& in_parallel_distribution =
        ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    CheckParallelDistribution(hierarchy, axis, in_parallel_distribution,
                              ctx->ParallelDistribution4ArgNameAndIndex("indices", 0),
                              ctx->ParallelDistribution4ArgNameAndIndex("out", 0));
    const TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
    TensorSliceView view =
        GetTensorSliceView4ParallelId(hierarchy, in_parallel_distribution, in_logical_desc->shape(),
                                      ctx->parallel_ctx().parallel_id());
    return std::make_shared<GatherOpKernelState>(view.At(axis).begin(), view.At(axis).end());
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t num_indices = indices->shape().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t offset = 0;
    if (state != nullptr) {
      auto* gather_state = dynamic_cast<GatherOpKernelState*>(state);
      CHECK_NOTNULL(gather_state);
      CHECK_EQ(in->shape().At(axis), gather_state->upper() - gather_state->lower());
      offset = gather_state->lower();
    }

    GatherKernelUtilImpl<device_type, T, K>::Forward(
        ctx->device_ctx(), indices->dptr<K>(), num_indices, in->dptr<T>(),
        GetFlatShape(in->shape(), axis), out->mut_dptr<T>(), offset);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GATHER_KERNEL(device, in_type, indices_type)                                \
  REGISTER_USER_KERNEL("gather")                                                             \
      .SetCreateFn<                                                                          \
          GatherKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                   \
                       & (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_type))       \
                       & (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_KERNEL, DEVICE_TYPE_SEQ, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
