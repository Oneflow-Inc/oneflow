#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

class AssignKernel final : public user_op::OpKernel {
 public:
  AssignKernel() = default;
  ~AssignKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* value_tensor = ctx->Tensor4ArgNameAndIndex("value", 0);
    user_op::Tensor* ref_tensor = ctx->Tensor4ArgNameAndIndex("ref", 0);
    if (value_tensor->dptr() == ref_tensor->dptr()) { return; }
    size_t tensor_bytes_size =
        ref_tensor->shape().elem_cnt() * GetSizeOfDataType(ref_tensor->data_type());
    size_t val_tensor_bytes_size =
        value_tensor->shape().elem_cnt() * GetSizeOfDataType(value_tensor->data_type());
    CHECK_EQ(tensor_bytes_size, val_tensor_bytes_size);
    AutoMemcpy(ctx->device_ctx(), ref_tensor->mut_dptr(), value_tensor->dptr(), tensor_bytes_size,
               ref_tensor->mem_case(), value_tensor->mem_case());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

}  // namespace

REGISTER_USER_KERNEL("assign").SetCreateFn<AssignKernel>().SetIsMatchedHob(user_op::HobTrue());

}  // namespace oneflow
