#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {
template<typename T>
class SqrtNpuKernel final : public user_op::OpKernel{
 public:
  SqrtNpuKernel() = default;
  ~SqrtNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t x_elem_cnt = x->shape_view().elem_cnt();
    const int64_t y_elem_cnt = y->shape_view().elem_cnt();

    if (x_elem_cnt != 0 && y_elem_cnt != 0) {
        NpuCommand npu_command;
        npu_command.OpName("Sqrt")
                   .Input(x)
                   .Output(y)
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run()
               .Realease();
        //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
        // PrintResult(z);
        // std::cout<<"Div Execute Over"<<std::endl; 
    } else {
      // For 0-d Tensor
      std::cout<<"0-d Tensor"<<std::endl;
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_SQRT_NPU_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("sqrt")                                                            \
      .SetCreateFn<SqrtNpuKernel<dtype>>()                                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));   
REGISTER_SQRT_NPU_KERNEL(float);
REGISTER_SQRT_NPU_KERNEL(float16);

}