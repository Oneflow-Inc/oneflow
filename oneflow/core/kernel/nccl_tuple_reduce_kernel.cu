#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "nccl.h"

namespace oneflow {

class NcclTupleReduceKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleReduceKernel);
  NcclTupleReduceKernel() = default;
  ~NcclTupleReduceKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  ParallelContext parallel_ctx_;
};

void NcclTupleReduceKernel::VirtualKernelInit(const ParallelContext* ctx) { parallel_ctx_ = *ctx; }

void NcclTupleReduceKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclTupleReduceOpConf& conf = this->op_conf().nccl_tuple_reduce_conf();
  NcclCheck(ncclGroupStart());
  FOR_RANGE(int64_t, i, 0, conf.out_size()) {
    const Blob* in = BnInOp2Blob(GenRepeatedBn("in", i));
    const void* send = in->dptr();
    Blob* out = BnInOp2Blob(GenRepeatedBn("out", i));
    void* recv = conf.root(i) == parallel_ctx_.rank_ctx().rank_id() ? out->mut_dptr() : nullptr;
    NcclCheck(ncclReduce(send, recv, in->shape().elem_cnt(), GetNcclDataType(in->data_type()),
                         ncclRedOp_t::ncclSum, conf.root(i), ctx.device_ctx->nccl_handle(),
                         ctx.device_ctx->cuda_stream()));
  }
  NcclCheck(ncclGroupEnd());
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kNcclTupleReduceConf, DeviceType::kGPU,
                            NcclTupleReduceKernel);

}  // namespace oneflow
