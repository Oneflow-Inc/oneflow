#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"

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
    const std::vector<int64_t> device_set_machine_ids =
        ctx->GetAttr<std::vector<int64_t>>("device_set_machine_ids");
    const std::vector<int64_t> device_set_device_ids =
        ctx->GetAttr<std::vector<int64_t>>("device_set_device_ids");
    CHECK_EQ(device_set_machine_ids.size(), device_set_device_ids.size());
    std::set<std::pair<int64_t, int64_t>> device_set;
    FOR_RANGE(int64_t, i, 0, device_set_machine_ids.size()) {
      device_set.emplace(std::make_pair(device_set_machine_ids.at(i), device_set_device_ids.at(i)));
    }
    ncclComm_t comm = Global<EagerNcclCommMgr>::Get()->GetCommForDevice(device_set);
    NcclCheck(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                            GetNcclDataType(in->data_type()), ncclSum, comm,
                            ctx->device_ctx()->cuda_stream()));
  };
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerNcclAllReduceKernel>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kGPU;
    });

}  // namespace oneflow
