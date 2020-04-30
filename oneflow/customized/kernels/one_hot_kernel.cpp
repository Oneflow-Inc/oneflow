#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
class CpuOneHotKernel final : public user_op::OpKernel {
 public:
  CpuOneHotKernel() = default;
  ~CpuOneHotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_indices = indices->shape().elem_cnt();
    const int64_t depth = ctx->GetAttr<int64_t>("depth");
    const float on_value = ctx->GetAttr<float>("on_value");
    const float off_value = ctx->GetAttr<float>("off_value");
    const T* indices_dptr = indices->dptr<T>();
    K* out_dptr = out->mut_dptr<K>();

    NewKernelUtil<DeviceType::kCPU>::Fill(ctx->device_ctx(), out->shape().elem_cnt(), off_value,
                                          out->mut_dptr<K>());
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices_dptr[i];
      CHECK_GE(idx, 0);
      CHECK_LT(idx, depth);
      out_dptr[i * depth + idx] = static_cast<K>(on_value);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ONE_HOT_KERNEL(idtype, odtype)                                                \
  REGISTER_USER_KERNEL("one_hot").SetCreateFn<CpuOneHotKernel<idtype, odtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                                   \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0);    \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);            \
        return ctx.device_type() == DeviceType::kCPU                                               \
               && indices_desc->data_type() == GetDataType<idtype>::value                          \
               && out_desc->data_type() == GetDataType<odtype>::value;                             \
      });

REGISTER_CPU_ONE_HOT_KERNEL(int32_t, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(int32_t, int64_t)
REGISTER_CPU_ONE_HOT_KERNEL(int32_t, float)
REGISTER_CPU_ONE_HOT_KERNEL(int32_t, double)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, int64_t)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, float)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, double)

}  // namespace oneflow
