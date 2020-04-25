#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/unsorted_batch_segment_sum_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class UnsortedBatchSegmentSumKernel final : public user_op::OpKernel {
 public:
  UnsortedBatchSegmentSumKernel() = default;
  ~UnsortedBatchSegmentSumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    const user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t num_segment_ids = segment_ids->shape().elem_cnt();
   BatchGatherKernelUtil<device_type, T>::Backward(ctx->device_ctx(), data->dptr<T>(),
       segment_ids->dptr<T>(), out->mut_dptr<T>());
  }
};

#define REGISTER_UNSORTED_BATCH_SEGMENT_SUM_KERNEL(device, T_dtype)                         \
  REGISTER_USER_KERNEL("unsorted_batch_segment_sum")                                         \
  .SetCreateFn<UnsortedBatchSegmentSumKernel<device, OF_PP_PAIR_FIRST(T_dtype)>>()            \
  .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
      const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
      return ctx.device_type() == device                                                      \
             && out_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype);                           \
  });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_BATCH_SEGMENT_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 UNSORTED_BATCH_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
