#include "oneflow/core/kernel/foreign_output_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

void ForeignOutputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_output_conf().ofblob_buffer_name();
  std::shared_ptr<ForeignJobInstance> foreign_job_instance;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) {
    OfBlob ofblob(ctx.device_ctx, BnInOp2Blob("in"));
    foreign_job_instance->PullBlob(reinterpret_cast<uint64_t>(&ofblob));
  }
}

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kForeignOutputConf, ForeignOutputKernel);

}  // namespace oneflow
