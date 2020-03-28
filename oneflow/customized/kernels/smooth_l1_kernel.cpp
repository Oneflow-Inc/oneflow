#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class SmoothL1CPUKernel final : public user_op::OpKernel {
 public:
  SmoothL1CPUKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}

  SmoothL1CPUKernel() = default;
  ~SmoothL1CPUKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override { return; };

  std::unique_ptr<std::mt19937> random_generator_;
};

#define REGISTER_SMOOTH_L1_CPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("smooth_l1")                                                          \
      .SetCreateFn(                                                                          \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1CPUKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                           \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);          \
        return ctx.device_type() == DeviceType::kCPU                                         \
               && y_desc->data_type() == GetDataType<dtype>::value;                          \
      });

REGISTER_SMOOTH_L1_CPU_KERNEL(float)
REGISTER_SMOOTH_L1_CPU_KERNEL(double)

}  // namespace oneflow
