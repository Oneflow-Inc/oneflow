#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class CopyDataContentKernel final : public user_op::OpKernel {
 public:
  CopyDataContentKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  CopyDataContentKernel() = default;
  ~CopyDataContentKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt());
    CHECK_EQ(in->data_type(), out->data_type());
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<void>(), in->dptr<void>(),
                        in->shape().elem_cnt() * GetSizeOfDataType(in->data_type()));
  };
};

}  // namespace oneflow
