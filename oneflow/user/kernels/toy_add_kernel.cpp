#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {

template<typename T>
class CpuToyAddKernel final : public user_op::OpKernel {
 public:
  CpuToyAddKernel() = default;
  ~CpuToyAddKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    const T* y_ptr = y->dptr<T>();
    T* out_ptr = output->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { out_ptr[i] = x_ptr[i] +y_ptr[i]; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_TOY_ADD_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("toy_add")                              \
      .SetCreateFn<CpuToyAddKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_TOY_ADD_KERNEL(float)
REGISTER_CPU_TOY_ADD_KERNEL(double)
REGISTER_CPU_TOY_ADD_KERNEL(uint8_t)
REGISTER_CPU_TOY_ADD_KERNEL(int8_t)
REGISTER_CPU_TOY_ADD_KERNEL(int32_t)
REGISTER_CPU_TOY_ADD_KERNEL(int64_t)

}  // namespace oneflow