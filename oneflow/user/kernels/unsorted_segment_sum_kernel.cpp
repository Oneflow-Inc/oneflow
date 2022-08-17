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
#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace oneflow {

namespace user_op {

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
    const TensorDesc* out_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("out", 0);
    TensorSliceView view = GetTensorSliceView4ParallelId(
        hierarchy, out_nd_sbp, out_logical_desc->shape(), ctx->parallel_ctx().parallel_id());
    return std::make_shared<UnsortedSegmentSumOpKernelCache>(view.At(axis).begin(),
                                                             view.At(axis).end());
  } else {
    return nullptr;
  }
}

}  // namespace

template<DeviceType device_type, typename T, typename K>
class UnsortedSegmentSumKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  UnsortedSegmentSumKernel() = default;
  ~UnsortedSegmentSumKernel() override = default;

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
    Memset<device_type>(ctx->stream(), out->mut_dptr(), 0,
                        out->shape_view().elem_cnt() * sizeof(T));

    int64_t offset = 0;
    if (cache != nullptr) {
      auto* sum_cache = dynamic_cast<const UnsortedSegmentSumOpKernelCache*>(cache);
      CHECK_NOTNULL(sum_cache);
      CHECK_EQ(out->shape_view().At(axis), sum_cache->upper() - sum_cache->lower());
      offset = sum_cache->lower();
    }

    if (num_segment_ids != 0) {
      UnsortedSegmentSumKernelUtil<device_type, T, K, T>::UnsortedSegmentSum(
          ctx->stream(), segment_ids->dptr<K>(), data->dptr<T>(), num_segment_ids, num_segments,
          outer_dim_size, inner_dim_size, offset, out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(device, out_type, segment_ids_type, kernel_type) \
  REGISTER_USER_KERNEL(kernel_type)                                                           \
      .SetCreateFn<UnsortedSegmentSumKernel<device, OF_PP_PAIR_FIRST(out_type),               \
                                            OF_PP_PAIR_FIRST(segment_ids_type)>>()            \
      .SetIsMatchedHob(                                                                       \
          (user_op::HobDeviceType() == device)                                                \
          && (user_op::HobDataType("segment_ids", 0) == OF_PP_PAIR_SECOND(segment_ids_type))  \
          && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_type)));

#define REGISTER_UNSORTED_SEGMENT_SUM_KERNEL_CASE(device_type, out_type, segment_ids_type) \
  REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(device_type, out_type, segment_ids_type,            \
                                       ("unsorted_segment_sum"))

#define REGISTER_UNSORTED_SEGMENT_SUM_LIKE_KERNEL_CASE(device_type, out_type, segment_ids_type) \
  REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(device_type, out_type, segment_ids_type,                 \
                                       ("unsorted_segment_sum_like"))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_SEGMENT_SUM_KERNEL_CASE, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_SEGMENT_SUM_LIKE_KERNEL_CASE, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
template<typename T, typename K>
class UnsortedSegmentSumHalfKernel final : public user_op::OpKernel {
 public:
  UnsortedSegmentSumHalfKernel() = default;
  ~UnsortedSegmentSumHalfKernel() override = default;

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
    user_op::Tensor* tmp_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t outer_dim_size = out->shape_view().Count(0, axis);
    int64_t num_segments = out->shape_view().At(axis);
    int64_t inner_dim_size = out->shape_view().Count(axis + 1);
    int64_t num_segment_ids = segment_ids->shape_view().elem_cnt();
    Memset<DeviceType::kCUDA>(ctx->stream(), tmp_buf->mut_dptr(), 0,
                              out->shape_view().elem_cnt() * sizeof(float));
    int64_t offset = 0;
    if (cache != nullptr) {
      auto* sum_cache = dynamic_cast<const UnsortedSegmentSumOpKernelCache*>(cache);
      CHECK_NOTNULL(sum_cache);
      CHECK_EQ(out->shape_view().At(axis), sum_cache->upper() - sum_cache->lower());
      offset = sum_cache->lower();
    }

    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, float, K, T>::UnsortedSegmentSum(
        ctx->stream(), segment_ids->dptr<K>(), data->dptr<T>(), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size, offset, tmp_buf->mut_dptr<float>());

    auto f2h = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
        ctx->device_type(), DataType::kFloat, out->data_type());
    CHECK(f2h);
    f2h->Launch(ctx->stream(), tmp_buf->dptr<float>(), out->mut_dptr<T>(),
                out->shape_view().elem_cnt());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_HALF_HALF_KERNEL(out_type, segment_ids_type, kernel_type) \
  REGISTER_USER_KERNEL(kernel_type)                                                             \
      .SetCreateFn<UnsortedSegmentSumHalfKernel<OF_PP_PAIR_FIRST(out_type),                     \
                                                OF_PP_PAIR_FIRST(segment_ids_type)>>()          \
      .SetIsMatchedHob(                                                                         \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                       \
          && (user_op::HobDataType("segment_ids", 0) == OF_PP_PAIR_SECOND(segment_ids_type))    \
          && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_type)))                   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                       \
        const Shape& out_shape = ctx->OutputShape("out", 0);                                    \
        return GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float));                        \
      });

#define REGISTER_UNSORTED_SEGMENT_SUM_HALF_KERNEL_CASE(out_type, segment_ids_type) \
  REGISTER_UNSORTED_SEGMENT_SUM_HALF_HALF_KERNEL(out_type, segment_ids_type,       \
                                                 ("unsorted_segment_sum"))         \
  REGISTER_UNSORTED_SEGMENT_SUM_HALF_HALF_KERNEL(out_type, segment_ids_type,       \
                                                 ("unsorted_segment_sum_like"))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_SEGMENT_SUM_HALF_KERNEL_CASE,
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#if CUDA_VERSION >= 11000
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_SEGMENT_SUM_HALF_KERNEL_CASE,
                                 OF_PP_MAKE_TUPLE_SEQ(nv_bfloat16, DataType::kBFloat16),
                                 INDEX_DATA_TYPE_SEQ)
#endif

#undef REGISTER_UNSORTED_SEGMENT_SUM_HALF_KERNEL_CASE

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
