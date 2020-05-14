#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class ConstantKernel final : public OpKernel {
 public:
  ConstantKernel() = default;
  ~ConstantKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->GetAttr<bool>("is_floating_value");
    const int64_t elem_cnt = out_tensor->shape().elem_cnt();
    CHECK(elem_cnt);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), elem_cnt,
                                     is_floating_value
                                         ? static_cast<T>(ctx->GetAttr<double>("floating_value"))
                                         : static_cast<T>(ctx->GetAttr<int64_t>("integer_value")),
                                     out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONSTANT_KERNEL(device, dtype)                                       \
  REGISTER_USER_KERNEL("constant")                                                    \
      .SetCreateFn<ConstantKernel<device, dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const auto& data_type = ctx.GetAttr<DataType>("dtype");                       \
        return ctx.device_type() == device && data_type == GetDataType<dtype>::value; \
      });

#define REGISTER_CONSTANT_KERNEL_WITH_DTYPE_PAIR(device, dtype_pair) \
  REGISTER_CONSTANT_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CONSTANT_KERNEL_WITH_DTYPE_PAIR, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
