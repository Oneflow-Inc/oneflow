#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

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
    int64_t num_segmentss = ctx->GetAttr<int64_t>("num_segments");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t outer_dim_size = out->shape().Count(0, axis);
    int64_t num_segments = out->shape().At(axis);
    int64_t inner_dim_size = out->shape().Count(axis + 1);
    int64_t num_segment_ids = segment_ids->shape().elem_cnt();

    UnsortedSegmentSumKernelUtil<device_type, T, K>::UnsortedSegmentSum(ctx->device_ctx(), segment_ids->dptr<K>(),
        data->dptr<T>(), num_segment_ids, num_segments, outer_dim_size, inner_dim_size, 0, out->mut_dptr<T>());
  }
};

#define REGISTER_UNSORTED_SEGMENT_SUM_KERNEL(device, T_dtype, K_dtype)                         \
  REGISTER_USER_KERNEL("unsorted_segment_sum")                                                 \
  .SetCreateFn<UnsortedSegmentSumKernel<device, OF_PP_PAIR_FIRST(T_dtype),                     \
                            OF_PP_PAIR_FIRST(K_dtype)>>()                                      \
  .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
      const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
      const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("segment_ids", 0);  \
      return ctx.device_type() == device                                                       \
             && indices_desc->data_type() == OF_PP_PAIR_SECOND(K_dtype)                        \
             && out_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype);                           \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_SEGMENT_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)


}  // namespace oneflow
