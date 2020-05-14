#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class CpuInvertPermutationKernel final : public user_op::OpKernel {
 public:
  CpuInvertPermutationKernel() = default;
  ~CpuInvertPermutationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t elem_cnt = in->shape().elem_cnt();
    const T* x_ptr = in->dptr<T>();
    T* y_ptr = out->mut_dptr<T>();
    std::fill_n(y_ptr, elem_cnt, -1);

    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      const T d = x_ptr[i];

      CHECK(d < elem_cnt && d >= 0) << d << " is not between 0 and " << elem_cnt;
      CHECK(y_ptr[d] == -1) << d << " is duplicated in the input.";

      y_ptr[d] = i;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_INVERT_PERMUTATION_KERNEL(dtype)                                 \
  REGISTER_USER_KERNEL("invert_permutation")                                          \
      .SetCreateFn<CpuInvertPermutationKernel<dtype>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && in_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_INVERT_PERMUTATION_KERNEL(int32_t)
REGISTER_CPU_INVERT_PERMUTATION_KERNEL(int64_t)

}  // namespace oneflow