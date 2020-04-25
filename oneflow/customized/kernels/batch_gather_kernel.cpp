#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/batch_gather_kernel.h"

namespace oneflow {

Shape getFlatShape(const ShapeView& shape, int64_t axis) {
    return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
class BatchGatherKernel final : public user_op::OpKernel {
 public:
  BatchGatherKernel() = default;
  ~BatchGatherKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t axis = ctx->GetAttr<int64_t>("axis");
    const int64_t batch_dims = ctx->GetAttr<int64_t>("batch_dims");
    const int64_t num_indices = indices->shape().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    BatchGatherKernel<device_type, T, K>::Forward(ctx->device_ctx(), in->dptr<T>(),
        indices->dptr<T>(), out->mut_dptr<T>());
  }
};

#define REGISTER_Batch_Gather_KERNEL(device, T_dtype)                                        \
  REGISTER_USER_KERNEL("batch_gather")                                                                \
  .SetCreateFn<BatchGatherKernel<device, OF_PP_PAIR_FIRST(T_dtype)>>()                         \
  .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                  \
      const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
      return ctx.device_type() == device                                                        \
             && out_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype);                            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BATCH_GATHER_KERNEL, DEVICE_TYPE_SEQ, BATCH_GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
