#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K, void (*f)(DeviceType &device_type, T &x , T &y)>
class BroadcastLogicalBinaryKernel final : public user_op::OpKernel {
 public:
  BroadcastLogicalBinaryKernel() = default;
  ~BroadcastLogicalBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);
    size_t num_axes = z->shape().NumAxes();

    f(device_type, x, y);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define LOGICAL_MATH_TYPE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ("Greater", GT)       \
  OF_PP_MAKE_TUPLE_SEQ("GreaterEqual", GE)  \
  OF_PP_MAKE_TUPLE_SEQ("LessThan", LT)      \
  OF_PP_MAKE_TUPLE_SEQ("LessEqual", LE)     \
  OF_PP_MAKE_TUPLE_SEQ("Equal", EQ)         \
  OF_PP_MAKE_TUPLE_SEQ("NotEqual", NE)      \
  OF_PP_MAKE_TUPLE_SEQ("LogicalAnd", AND)


#define REGISTER_BROADCAST_BINARY_KERNEL(logical_math_type, device, T_dtype, K_dtype)                        \
  REGISTER_USER_KERNEL("broadcast_binary")                                                                   \
  .SetCreateFn<BroadcastLogicalBinaryKernel<device, OF_PP_PAIR_FIRST(T_dtype), OF_PP_PAIR_FIRST(K_dtype),                                                               \
      &NdarrayUtil<device, OF_PP_PAIR_FIRST(T_dtype)>::BroadcastGT                           \
           >>()                                  \
  .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                               \
      const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);                            \
      const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);                            \
      const std::string binary_math_type = ctx.GetAttr("binary_math_type");                                   \
      const user_op::TensorDesc* z_desc = ctx.TensorDesc4ArgNameAndIndex("z", 0);                                  \
      return ctx.device_type() == device                                                                     \
             && x_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype)                                            \
             && y_desc->data_type() == OF_PP_PAIR_SECOND(T_dtype)                                            \
             && binary_math_type == OF_PP_PAIR_FIRST(logical_math_type)                                     \
             && z_desc->data_type() == OF_PP_PAIR_SECOND(K_dtype);                                           \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_BINARY_KERNEL, LOGICAL_MATH_TYPE_SEQ,
    DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
