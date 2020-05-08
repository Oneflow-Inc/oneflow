#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T, typename K>
class UnsortedSegmentSumKernel final : public user_op::OpKernel {
 public:
  UnsortedSegmentSumKernel() = default;
  ~UnsortedSegmentSumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    const user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    int64_t axis = ctx->GetAttr<int64_t>("axis");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t outer_dim_size = out->shape().Count(0, axis);
    int64_t num_segments = out->shape().At(axis);
    int64_t inner_dim_size = out->shape().Count(axis + 1);
    int64_t num_segment_ids = segment_ids->shape().elem_cnt();
    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0, out->shape().elem_cnt() * sizeof(T));
    UnsortedSegmentSumKernelUtil<device_type, T, K>::UnsortedSegmentSum(
        ctx->device_ctx(), segment_ids->dptr<K>(), data->dptr<T>(), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size, 0, out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(device, out_type, segment_ids_type, kernel_type) \
  REGISTER_USER_KERNEL(kernel_type)                                                           \
      .SetCreateFn<UnsortedSegmentSumKernel<device, OF_PP_PAIR_FIRST(out_type),               \
                                            OF_PP_PAIR_FIRST(segment_ids_type)>>()            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                            \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);       \
        const user_op::TensorDesc* segment_ids_desc =                                         \
            ctx.TensorDesc4ArgNameAndIndex("segment_ids", 0);                                 \
        return ctx.device_type() == device                                                    \
               && segment_ids_desc->data_type() == OF_PP_PAIR_SECOND(segment_ids_type)        \
               && out_desc->data_type() == OF_PP_PAIR_SECOND(out_type);                       \
      });

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

}  // namespace user_op

}  // namespace oneflow
