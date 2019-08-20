#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "nccl.h"

namespace oneflow {

class NcclTupleBroadcastKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastKernel);
  NcclTupleBroadcastKernel() = default;
  ~NcclTupleBroadcastKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  ParallelContext parallel_ctx_;
};

void NcclTupleBroadcastKernel::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_ctx_ = *ctx;
}

void NcclTupleBroadcastKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclTupleBroadcastOpConf& conf = this->op_conf().nccl_tuple_broadcast_conf();
  NcclCheck(ncclGroupStart());
  FOR_RANGE(int64_t, i, 0, conf.out_size()) {
    const void* send = conf.root(i) == parallel_ctx_.rank_ctx().rank_id()
                           ? BnInOp2Blob(GenRepeatedBn("in", i))->dptr()
                           : nullptr;
    Blob* out = BnInOp2Blob(GenRepeatedBn("out", i));
    void* recv = out->mut_dptr();
    NcclCheck(ncclBroadcast(send, recv, out->shape().elem_cnt(), GetNcclDataType(out->data_type()),
                            conf.root(i), ctx.device_ctx->nccl_handle(),
                            ctx.device_ctx->cuda_stream()));
  }
  NcclCheck(ncclGroupEnd());
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kNcclTupleBroadcastConf, DeviceType::kGPU,
                            NcclTupleBroadcastKernel);

}  // namespace oneflow
