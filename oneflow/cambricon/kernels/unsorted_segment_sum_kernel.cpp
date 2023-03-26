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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

namespace {

void CheckNdSbp(const Shape& hierarchy, int64_t sum_axis, const NdSbp& segment_ids_nd_sbp,
                const NdSbp& data_nd_sbp, const NdSbp& out_nd_sbp) {
  CHECK_EQ(hierarchy.NumAxes(), segment_ids_nd_sbp.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), data_nd_sbp.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), out_nd_sbp.sbp_parallel_size());
  if (hierarchy.elem_cnt() == 1) { return; }
  FOR_RANGE(int64_t, i, 0, hierarchy.NumAxes()) {
    const auto& out_sbp = out_nd_sbp.sbp_parallel(i);
    if (out_sbp.has_split_parallel() && out_sbp.split_parallel().axis() == sum_axis) {
      CHECK(segment_ids_nd_sbp.sbp_parallel(i).has_broadcast_parallel());
      CHECK(data_nd_sbp.sbp_parallel(i).has_broadcast_parallel());
    }
  }
}

class UnsortedSegmentSumOpKernelCache final : public user_op::OpKernelCache {
 public:
  UnsortedSegmentSumOpKernelCache(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~UnsortedSegmentSumOpKernelCache() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

std::shared_ptr<user_op::OpKernelCache> CreateUnsortedSegmentSumOpKernelCache(
    user_op::KernelCacheContext* ctx) {
  if (ctx->parallel_ctx().parallel_num() > 1) {
    const auto axis = ctx->Attr<int64_t>("axis");
    const NdSbp& out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    CheckNdSbp(hierarchy, axis, ctx->NdSbp4ArgNameAndIndex("segment_ids", 0),
               ctx->NdSbp4ArgNameAndIndex("data", 0), out_nd_sbp);
    const user_op::TensorDesc* out_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("out", 0);
    TensorSliceView view = GetTensorSliceView4ParallelId(
        hierarchy, out_nd_sbp, out_logical_desc->shape(), ctx->parallel_ctx().parallel_id());
    return std::make_shared<UnsortedSegmentSumOpKernelCache>(view.At(axis).begin(),
                                                             view.At(axis).end());
  } else {
    return nullptr;
  }
}

}  // namespace

template<typename T>
class MluUnsortedSegmentSumKernel final : public user_op::OpKernel {
 public:
  MluUnsortedSegmentSumKernel() = default;
  ~MluUnsortedSegmentSumKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateUnsortedSegmentSumOpKernelCache(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    const user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    int64_t axis = ctx->Attr<int64_t>("axis");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t outer_dim_size = out->shape_view().Count(0, axis);
    int64_t num_segments = out->shape_view().At(axis);
    int64_t inner_dim_size = out->shape_view().Count(axis + 1);
    int64_t num_segment_ids = segment_ids->shape_view().elem_cnt();

    Memset<DeviceType::kMLU>(ctx->stream(), out->mut_dptr(), 0,
                             out->shape_view().elem_cnt() * sizeof(T));

    int64_t offset = 0;
    if (cache != nullptr) {
      auto* sum_cache = dynamic_cast<const UnsortedSegmentSumOpKernelCache*>(cache);
      CHECK_NOTNULL(sum_cache);
      CHECK_EQ(out->shape_view().At(axis), sum_cache->upper() - sum_cache->lower());
      offset = sum_cache->lower();
    }

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());
    if constexpr (std::is_same<T, float16>::value) {
      bang_unsorted_segment_sum_half_kernel<int32_t>(
          handle, data->dptr<T>(), outer_dim_size, num_segments, inner_dim_size,
          segment_ids->dptr<int32_t>(), num_segment_ids, out->mut_dptr<T>(), offset);
    } else {
      bang_unsorted_segment_sum_kernel<T, int32_t>(
          handle, data->dptr<T>(), outer_dim_size, num_segments, inner_dim_size,
          segment_ids->dptr<int32_t>(), num_segment_ids, out->mut_dptr<T>(), offset);
    }
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
