#include "oneflow/core/kernel/wait_and_send_ids_kernel.h"

namespace oneflow {

template<typename T>
void WaitAndSendIdsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto* status = static_cast<WaitAndSendIdsStatus*>(ctx.other);
  const auto& conf = this->op_conf().wait_and_send_ids_conf();
  if (status->out_idx_ >= status->out_num_) {
    status->buffer_status_ =
        Global<BufferMgr<int64_t>>::Get()->Get(conf.wait_buffer_name())->Receive(&status->in_id_);
    if (status->buffer_status_ == kBufferStatusErrorClosed) { return; }
    status->out_idx_ = 0;
    status->out_num_ = conf.id_list(status->in_id_).value_size();
  }
  *BnInOp2Blob("out")->mut_dptr<T>() = conf.id_list(status->in_id_).value(status->out_idx_);
  ++status->out_idx_;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kWaitAndSendIdsConf, WaitAndSendIdsKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
