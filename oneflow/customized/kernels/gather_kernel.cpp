#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

}  // namespace

template<DeviceType device_type, typename T, typename K>
class GatherKernel final : public user_op::OpKernel {
 public:
  GatherKernel() = default;
  ~GatherKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t axis = ctx->GetAttr<int64_t>("axis");
    const int64_t num_indices = indices->shape().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    GatherKernelUtilImpl<device_type, T, K>::Forward(
        ctx->device_ctx(), indices->dptr<K>(), num_indices, in->dptr<T>(),
        GetFlatShape(in->shape(), axis), out->mut_dptr<T>(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GATHER_KERNEL(device, in_type, indices_type)                                   \
  REGISTER_USER_KERNEL("gather")                                                                \
      .SetCreateFn<                                                                             \
          GatherKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>()    \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0);           \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        return ctx.device_type() == device                                                      \
               && indices_desc->data_type() == OF_PP_PAIR_SECOND(indices_type)                  \
               && in_desc->data_type() == OF_PP_PAIR_SECOND(in_type);                           \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_KERNEL, DEVICE_TYPE_SEQ, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
