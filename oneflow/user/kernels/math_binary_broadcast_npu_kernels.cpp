
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

class MathBinaryBroadcastAddKernel final : public user_op::OpKernel{
 public:
  MathBinaryBroadcastAddKernel() = default;
  ~MathBinaryBroadcastAddKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    const int64_t x_elem_cnt = x->shape().elem_cnt();
    const int64_t y_elem_cnt = y->shape().elem_cnt();

    if (x_elem_cnt != 0 && y_elem_cnt != 0) {
        NpuCommand npu_command;
        npu_command.OpName("Add")
                   .Input(x,"channels_nd")
                   .Input(y,"channels_nd")
                   .Output(z,"channels_nd")
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run();
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
        //PrintResult(out_tensor);
        //std::cout<<"MathBinaryBroadcastAddKernel Execute Over"<<std::endl; 
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BROADCASTADD_NPU_KERNEL(math_type_pair)        \
        REGISTER_USER_KERNEL(math_type_pair)                    \
        .SetCreateFn<MathBinaryBroadcastAddKernel>()            \
        .SetIsMatchedHob( user_op::HobDeviceType() == DeviceType::kNPU );
REGISTER_BROADCASTADD_NPU_KERNEL("broadcast_add")

class MathBinaryBroadcastDivKernel final : public user_op::OpKernel{
 public:
  MathBinaryBroadcastDivKernel() = default;
  ~MathBinaryBroadcastDivKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const int64_t x_elem_cnt = x->shape().elem_cnt();
    const int64_t y_elem_cnt = y->shape().elem_cnt();

    if (x_elem_cnt != 0 && y_elem_cnt != 0) {
        NpuCommand npu_command;
        npu_command.OpName("RealDiv")
                   .Input(x)
                   .Input(y)
                   .Output(z)
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run();
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
        PrintResult(z);
        std::cout<<"Div Execute Over"<<std::endl; 
    } else {
      // For 0-d Tensor
      std::cout<<"0-d Tensor"<<std::endl;
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_BROADCASTDIV_NPU_KERNEL(math_type_pair)        \
        REGISTER_USER_KERNEL(math_type_pair)                    \
        .SetCreateFn<MathBinaryBroadcastDivKernel>()            \
        .SetIsMatchedHob( user_op::HobDeviceType() == DeviceType::kNPU );
REGISTER_BROADCASTDIV_NPU_KERNEL("broadcast_div")

} // oneflow