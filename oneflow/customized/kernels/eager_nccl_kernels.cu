#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

class EagerNcclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerNcclAllReduceKernel() = default;
  ~EagerNcclAllReduceKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    ncclComm_t comm;
    NcclCheck(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                            GetNcclDataType(in->data_type()), ncclSum, comm,
                            ctx->device_ctx()->cuda_stream()));
  };
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerNcclAllReduceKernel>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

}  // namespace oneflow
