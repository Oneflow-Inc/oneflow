#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/preprocessor.h"
namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T, typename K,
         void (*binary_func)(DeviceCtx* ctx, const XpuVarNdarray<int8_t>& z,
                             const XpuVarNdarray<const T>& x, const XpuVarNdarray<const T>& y)>
class BroadcastLogicalBinaryKernel final : public user_op::OpKernel {
 public:
  BroadcastLogicalBinaryKernel() = default;
  ~BroadcastLogicalBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("out", 0);
    const XpuVarNdarray<int8_t>& ndarray_z =
        XpuVarNdarray<int8_t>(z->shape(), z->mut_dptr<int8_t>());
    const XpuVarNdarray<const T>& ndarray_x = XpuVarNdarray<const T>(x->shape(), x->dptr<T>());
    const XpuVarNdarray<const T>& ndarray_y = XpuVarNdarray<const T>(y->shape(), y->dptr<T>());
    binary_func(ctx->device_ctx(), ndarray_z, ndarray_x, ndarray_y);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define LOGICAL_OP_TYPE_NAME_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater", GT)       \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_greater_equal", GE) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less", LT)          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_less_equal", LE)    \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_equal", EQ)         \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_not_equal", NE)     \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_logical_and", AND)

#define REGISTER_BROADCAST_BINARY_KERNEL(logical_math_type, device, T_dtype, K_dtype)              \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(logical_math_type))                                        \
      .SetCreateFn<                                                                                \
          BroadcastLogicalBinaryKernel<device, OF_PP_PAIR_FIRST(T_dtype), int8_t,                  \
                                       &NdarrayUtil<device, OF_PP_PAIR_FIRST(T_dtype)>::OF_PP_CAT( \
                                           Broadcast, OF_PP_PAIR_SECOND(logical_math_type))>>()    \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("a", 0);                \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("b", 0);                \
        const user_op::TensorDesc* z_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);              \
        return ctx.device_type() == device && x_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype)    \
               && y_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype)                                \
               && z_desc->data_type() == DataType::kInt8;                                          \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_BINARY_KERNEL, LOGICAL_OP_TYPE_NAME_SEQ,
                                 DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
