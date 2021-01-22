#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_CPU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_CPU_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<DeviceType device_type, typename FunctorT, typename T>
class UnaryElemwiseCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryElemwiseCpuKernel);
  UnaryElemwiseCpuKernel(const std::string& input_name, const std::string& output_name,
                      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn)
      : input_name(input_name), output_name(output_name), FunctorCreateFn(FunctorCreateFn) {}

  std::string input_name = "in";    // The name for the input tensor
  std::string output_name = "out";  // The name for the output tensor

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex(input_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();

    FunctorT functor = FunctorCreateFn(ctx);
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      out_ptr[i] = functor(in_ptr[i]);
    }
  
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


}  // namespace oneflow
#endif // _ONEFLOW_USER_KERNELS_ELEMENTWISE_CPU_KERNEL_H_
