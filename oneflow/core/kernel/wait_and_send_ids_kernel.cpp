#include "oneflow/core/kernel/wait_and_send_ids_kernel.h"

namespace oneflow {

void WaitAndSendIdsKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto* status = static_cast<WaitAndSendIdsStatus*>(ctx.other);
  const auto& conf = op_conf().wait_and_send_ids_conf();
  if (status->out_idx_ >= status->out_num_) {
    status->channel_status_ =
        Global<BufferMgr<int32_t>>::Get()->Get(conf.wait_channel_name())->Receive(&status->in_id_);
    if (status->channel_status_ == kChannelStatusErrorClosed) { return; }
    status->out_idx_ = 0;
    status->out_num_ = conf.id_list(status->in_id_).id_size();
  }
  *BnInOp2Blob("out")->mut_dptr<int32_t>() = conf.id_list(status->in_id_).id(status->out_idx_);
  ++status->out_idx_;
}

REGISTER_KERNEL(OperatorConf::kWaitAndSendIdsConf, WaitAndSendIdsKernel);

}  // namespace oneflow
