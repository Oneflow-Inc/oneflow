#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/batch_gather_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T, typename K>
class BatchGatherKernel final : public user_op::OpKernel {
 public:
  BatchGatherKernel() = default;
  ~BatchGatherKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = indices->shape().NumAxes() - 1;
    const Shape flat_out_shape =
        Shape({out->shape().Count(0, axis), out->shape().At(axis), out->shape().Count(axis + 1)});
    BatchGatherKernelUtilImpl<device_type, T, K>::Forward(ctx->device_ctx(), in->dptr<T>(),
                                                          indices->dptr<K>(), flat_out_shape,
                                                          in->shape().At(axis), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_GATHER_KERNEL(device, out_dtype, indices_dtype)                          \
  REGISTER_USER_KERNEL("batch_gather")                                                          \
      .SetCreateFn<BatchGatherKernel<device, OF_PP_PAIR_FIRST(out_dtype),                       \
                                     OF_PP_PAIR_FIRST(indices_dtype)>>()                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        return ctx.device_type() == device                                                      \
               && indices_desc->data_type() == OF_PP_PAIR_SECOND(indices_dtype)                 \
               && out_desc->data_type() == OF_PP_PAIR_SECOND(out_dtype);                        \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BATCH_GATHER_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
