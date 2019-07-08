#include "oneflow/core/kernel/foreign_input_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {

void ForeignInputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_input_conf().ofblob_buffer_name();
  std::shared_ptr<JobInstance> job_instance;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) {
    OfBlob ofblob(ctx.device_ctx, BnInOp2Blob("out"));
    job_instance->PushBlob(reinterpret_cast<uint64_t>(&ofblob));
  }
}

REGISTER_KERNEL(OperatorConf::kForeignInputConf, ForeignInputKernel);

}  // namespace oneflow
