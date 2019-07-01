#include "oneflow/core/kernel/foreign_output_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

void ForeignOutputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_output_conf().ofblob_buffer_name();
  std::shared_ptr<ForeignCallback> foreign_callback;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignCallback>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_callback);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  OfBlob ofblob(ctx.device_ctx, BnInOp2Blob("in"));
  foreign_callback->PullBlob(reinterpret_cast<uint64_t>(&ofblob));
}

REGISTER_KERNEL(OperatorConf::kForeignOutputConf, ForeignOutputKernel);

}  // namespace oneflow
