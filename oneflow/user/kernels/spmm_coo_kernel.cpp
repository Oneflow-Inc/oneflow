#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class CpuSpmmCooKernel final : public user_op::OpKernel {
 public:
  CpuSpmmCooKernel() = default;
  ~CpuSpmmCooKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CHECK_EQ(1, 2) << "spmm for cpu is not implemented yet ";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SPMM_COO_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("spmm_coo")                              \
      .SetCreateFn<CpuSpmmCooKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SPMM_COO_KERNEL(float)
REGISTER_CPU_SPMM_COO_KERNEL(double)
REGISTER_CPU_SPMM_COO_KERNEL(uint8_t)
REGISTER_CPU_SPMM_COO_KERNEL(int8_t)
REGISTER_CPU_SPMM_COO_KERNEL(int32_t)
REGISTER_CPU_SPMM_COO_KERNEL(int64_t)

}  // namespace oneflow