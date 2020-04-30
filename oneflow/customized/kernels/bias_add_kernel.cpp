#include "oneflow/customized/kernels/bias_add_kernel.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BiasAddUserKernel final : public user_op::OpKernel {
 public:
  BiasAddUserKernel() = default;
  ~BiasAddUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_blob = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_blob = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->GetAttr<int32_t>("axis");
    const int64_t outer_size = a_blob->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_blob->shape().At(bias_add_axis);
    const int64_t inner_size = a_blob->shape().Count(bias_add_axis + 1);
    BiasAddUtil<device_type, T>::BiasAdd(ctx->device_ctx(), outer_size, bias_size, inner_size,
                                         a_blob->dptr<T>(), b_blob->dptr<T>(),
                                         out_blob->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

};

template<typename T>
struct BiasAddUtil<DeviceType::kCPU, T> {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y) {
    const Shape in_out_shape({outer_size, bias_size, inner_size});
    const Shape bias_shape({1, bias_size, 1});
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastAdd(ctx, XpuVarNdarray<T>(in_out_shape, y),
                                                   XpuVarNdarray<const T>(in_out_shape, x),
                                                   XpuVarNdarray<const T>(bias_shape, bias));
  }
};

#define INSTANTIATE_BIAS_ADD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct BiasAddUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BIAS_ADD_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

#define REGISTER_KERNEL(op_device_type, dtype)                                                  \
  REGISTER_USER_KERNEL("bias_add")                                                              \
      .SetCreateFn<BiasAddUserKernel<DeviceType::k##op_device_type, dtype>>()                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == DeviceType::k##op_device_type                               \
               && out_desc->data_type() == GetDataType<dtype>::value;                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL(CPU, float16)
REGISTER_KERNEL(CPU, float)
REGISTER_KERNEL(CPU, double)
REGISTER_KERNEL(CPU, int8_t)
REGISTER_KERNEL(CPU, int32_t)
REGISTER_KERNEL(CPU, int64_t)

REGISTER_KERNEL(GPU, float16)
REGISTER_KERNEL(GPU, float)
REGISTER_KERNEL(GPU, double)
REGISTER_KERNEL(GPU, int8_t)
REGISTER_KERNEL(GPU, int32_t)
REGISTER_KERNEL(GPU, int64_t)

}  // namespace oneflow
