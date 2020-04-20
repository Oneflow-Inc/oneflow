#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

Shape getFlatShape(const ShapeView& shape, int64_t axis) {
    return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
class GatherKernel final : public user_op::OpKernel {
 public:
  GatherKernel() = default;
  ~GatherKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    int64_t axis = ctx->GetAttr<int64_t>("axis");
    int64_t batch_dims = ctx->GetAttr<int64_t>("batch_dims");
    int64_t num_indices = indices->shape().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    GatherKernelUtilImpl<device_type, T, K>::Forward(ctx->device_ctx(), indices->dptr<K>(),
       num_indices, in->dptr<T>(), getFlatShape(in->shape(), axis), out->dptr<T>(), 0);
  }
};

#define REGISTER_GATHER_KERNEL(device, T_dtype, K_dtype)                                        \
  REGISTER_USER_KERNEL("gather")                                                                \
  .SetCreateFn<GatherKernel<device, OF_PP_PAIR_FIRST(T_dtype),                                  \
                            OF_PP_PAIR_FIRST(K_dtype)>>()                                       \
  .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                  \
      const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
      const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0);   \
      return ctx.device_type() == device                                                        \
             && indices_desc->data_type() == OF_PP_PAIR_SECOND(K_dtype)                         \
             && out_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype);                            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_KERNEL, DEVICE_TYPE_SEQ, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
