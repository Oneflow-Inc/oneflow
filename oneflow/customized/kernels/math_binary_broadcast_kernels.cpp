#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/customized/ops/math_binary_broadcast_seq.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K,
         void (*binary_func)(DeviceCtx* ctx, const XpuVarNdarray<K>& z,
                             const XpuVarNdarray<const T>& x, const XpuVarNdarray<const T>& y)>
class MathBinaryBroadcastKernel final : public user_op::OpKernel {
 public:
  MathBinaryBroadcastKernel() = default;
  ~MathBinaryBroadcastKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    T* dptr_x = tensor_x->mut_dptr<T>();
    T* dptr_y = tensor_y->mut_dptr<T>();
    K* dptr_z = tensor_z->mut_dptr<K>();
    size_t num_axes = tensor_z->shape().NumAxes();
    binary_func(ctx->device_ctx(), XpuVarNdarray<K>(tensor_z->shape(), dptr_z, num_axes),
                XpuVarNdarray<const T>(tensor_x->shape(), dptr_x, num_axes),
                XpuVarNdarray<const T>(tensor_y->shape(), dptr_y, num_axes));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_BINARY_BROADCAST_KERNEL(math_type_pair, device, data_type_pair)      \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                   \
      .SetCreateFn<MathBinaryBroadcastKernel<                                              \
          device, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair),      \
          &NdarrayUtil<device, OF_PP_PAIR_FIRST(data_type_pair)>::OF_PP_CAT(               \
              Broadcast, OF_PP_PAIR_SECOND(math_type_pair))>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                         \
        const user_op::TensorDesc* tensor_desc_z = ctx.TensorDesc4ArgNameAndIndex("z", 0); \
        return ctx.device_type() == device                                                 \
               && tensor_desc_z->data_type() == OF_PP_PAIR_SECOND(data_type_pair);         \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                                 MATH_BINARY_BROADCAST_FUNC_SEQ, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)
// gpu half
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                                 MATH_BINARY_BROADCAST_FUNC_SEQ, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ)

#define REGISTER_MATH_BINARY_BROADCAST_LOGICAL_KERNEL(math_type_pair, device, data_type_pair) \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                      \
      .SetCreateFn<MathBinaryBroadcastKernel<                                                 \
          device, OF_PP_PAIR_FIRST(data_type_pair), int8_t,                                   \
          &NdarrayUtil<device, OF_PP_PAIR_FIRST(data_type_pair)>::OF_PP_CAT(                  \
              Broadcast, OF_PP_PAIR_SECOND(math_type_pair))>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                            \
        const user_op::TensorDesc* tensor_desc_x = ctx.TensorDesc4ArgNameAndIndex("x", 0);    \
        const user_op::TensorDesc* tensor_desc_z = ctx.TensorDesc4ArgNameAndIndex("z", 0);    \
        return ctx.device_type() == device                                                    \
               && tensor_desc_x->data_type() == OF_PP_PAIR_SECOND(data_type_pair)             \
               && tensor_desc_z->data_type() == DataType::kInt8;                              \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_LOGICAL_KERNEL,
                                 MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
