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
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/kernels/gather_kernel_util.h"

namespace oneflow {
namespace {

class GatherOpKernelCache final : public user_op::OpKernelCache {
 public:
  GatherOpKernelCache(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~GatherOpKernelCache() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

void CheckNdSbp(const Shape& hierarchy, int64_t gather_axis, const NdSbp& in_nd_sbp,
                const NdSbp& indices_nd_sbp, const NdSbp& out_nd_sbp) {
  CHECK_EQ(hierarchy.NumAxes(), in_nd_sbp.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), indices_nd_sbp.sbp_parallel_size());
  CHECK_EQ(hierarchy.NumAxes(), out_nd_sbp.sbp_parallel_size());
  if (hierarchy.elem_cnt() == 1) { return; }
  FOR_RANGE(int64_t, i, 0, hierarchy.NumAxes()) {
    const auto& in_sbp = in_nd_sbp.sbp_parallel(i);
    if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == gather_axis) {
      CHECK(indices_nd_sbp.sbp_parallel(i).has_broadcast_parallel());
      CHECK(out_nd_sbp.sbp_parallel(i).has_partial_sum_parallel());
    }
  }
}

template<typename T, typename K>
class MluGatherKernel final : public user_op::OpKernel {
 public:
  MluGatherKernel() = default;
  ~MluGatherKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    if (ctx->parallel_ctx().parallel_num() > 1) {
      const auto axis = ctx->Attr<int64_t>("axis");
      const NdSbp& in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      CheckNdSbp(hierarchy, axis, in_nd_sbp, ctx->NdSbp4ArgNameAndIndex("indices", 0),
                 ctx->NdSbp4ArgNameAndIndex("out", 0));
      const Shape in_logical_shape =
          ExpandDimIf0D(ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0)->shape());
      TensorSliceView view = GetTensorSliceView4ParallelId(hierarchy, in_nd_sbp, in_logical_shape,
                                                           ctx->parallel_ctx().parallel_id());
      return std::make_shared<GatherOpKernelCache>(view.At(axis).begin(), view.At(axis).end());
    } else {
      return nullptr;
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }

    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const Shape in_shape = ExpandDimIf0D(in->shape_view());

    const int64_t num_indices = indices->shape_view().elem_cnt();
    int64_t offset = 0;
    if (cache != nullptr) {
      auto* gather_cache = dynamic_cast<const GatherOpKernelCache*>(cache);
      CHECK_NOTNULL(gather_cache);
      CHECK_EQ(in_shape.At(axis), gather_cache->upper() - gather_cache->lower());
      offset = gather_cache->lower();
    }

    int64_t batch = in_shape.Count(0, axis);
    int64_t N = in_shape.At(axis);
    int64_t length = in_shape.Count(axis + 1);

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());

    if constexpr (std::is_same<T, float16>::value) {
      bang_gather_half_kernel<K>(handle, in->dptr<T>(), batch, N, length, indices->dptr<K>(),
                                 num_indices, out->mut_dptr<T>(), offset);
    } else {
      bang_gather_kernel<T, K>(handle, in->dptr<T>(), batch, N, length, indices->dptr<K>(),
                               num_indices, out->mut_dptr<T>(), offset);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_GATHER_KERNEL(in_type, indices_type)                                        \
  REGISTER_USER_KERNEL("gather")                                                                 \
      .SetCreateFn<MluGatherKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kMLU)                                         \
          && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_type))                       \
          && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));

#define BANG_GATHER_DATA_TYPE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)     \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(bool, DataType::kBool)       \
  OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16)

#define BANG_INDEX_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MLU_GATHER_KERNEL, BANG_GATHER_DATA_TYPE_SEQ,
                                 BANG_INDEX_DATA_TYPE_SEQ)

#undef BANG_GATHER_DATA_TYPE_SEQ
#undef BANG_INDEX_DATA_TYPE_SEQ
#undef REGISTER_MLU_GATHER_KERNEL

}  // namespace
}  // namespace oneflow
