#include "oneflow/customized/kernels/where_kernel.h"

namespace oneflow {

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    DoWhere(elem_cnt, cond, lhs, rhs, out);
  }
};

template<DeviceType device_type, typename T, typename CondT>
class WhereKernel final : public user_op::OpKernel {
 public:
  WhereKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    WhereFunctor<device_type, T, CondT>()(ctx->device_ctx(), out->shape().elem_cnt(),
                                          cond->dptr<CondT>(), x->dptr<T>(), y->dptr<T>(),
                                          out->mut_dptr<T>());
  }
};

#define REGISTER_WHERE_KERNEL(device_type_v, dtype_pair, ctype_pair)                           \
  REGISTER_USER_KERNEL("where")                                                                \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                       \
        return new WhereKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                    \
                               OF_PP_PAIR_FIRST(ctype_pair)>(ctx);                             \
      })                                                                                       \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                             \
        const user_op::TensorDesc* cond_desc = ctx.TensorDesc4ArgNameAndIndex("condition", 0); \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);        \
        return ctx.device_type() == device_type_v                                              \
               && cond_desc->data_type() == OF_PP_PAIR_SECOND(ctype_pair)                      \
               && out_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                      \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
