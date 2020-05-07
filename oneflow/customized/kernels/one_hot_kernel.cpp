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
    const DataType dtype = ctx->GetAttr<DataType>("dtype");
    const T on_value = IsFloatingDataType(dtype)
                           ? static_cast<T>(ctx->GetAttr<double>("floating_on_value"))
                           : static_cast<T>(ctx->GetAttr<int64_t>("integer_on_value"));
    const T off_value = IsFloatingDataType(dtype)
                            ? static_cast<T>(ctx->GetAttr<double>("floating_off_value"))
                            : static_cast<T>(ctx->GetAttr<int64_t>("integer_off_value"));
    const K* indices_dptr = indices->dptr<K>();
    T* out_dptr = out->mut_dptr<T>();

    NewKernelUtil<DeviceType::kCPU>::Fill(ctx->device_ctx(), out->shape().elem_cnt(), off_value,
                                          out->mut_dptr<T>());
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices_dptr[i];
      CHECK_GE(idx, 0);
      CHECK_LT(idx, depth);
      out_dptr[i * depth + idx] = on_value;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ONE_HOT_KERNEL(dtype, itype)                                                \
  REGISTER_USER_KERNEL("one_hot").SetCreateFn<CpuOneHotKernel<dtype, itype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                                 \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0);  \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
        return ctx.device_type() == DeviceType::kCPU                                             \
               && out_desc->data_type() == GetDataType<dtype>::value                             \
               && indices_desc->data_type() == GetDataType<itype>::value;                        \
      });

REGISTER_CPU_ONE_HOT_KERNEL(int32_t, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(int32_t, int64_t)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(int64_t, int64_t)
REGISTER_CPU_ONE_HOT_KERNEL(float, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(float, int64_t)
REGISTER_CPU_ONE_HOT_KERNEL(double, int32_t)
REGISTER_CPU_ONE_HOT_KERNEL(double, int64_t)

}  // namespace oneflow
