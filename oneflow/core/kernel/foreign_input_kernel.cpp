#include "oneflow/core/kernel/foreign_input_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/foreign_blob.h"
#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

void ForeignInputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_input_conf().foreign_blob_buffer_name();
  std::shared_ptr<ForeignCallback> foreign_callback;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignCallback>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_callback);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  ForeignBlob foreign_blob(ctx.device_ctx, BnInOp2Blob("out"));
  foreign_callback->PushBlob(foreign_blob);
}

REGISTER_KERNEL(OperatorConf::kForeignInputConf, ForeignInputKernel);

}  // namespace oneflow
