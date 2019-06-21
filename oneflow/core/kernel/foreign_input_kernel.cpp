#include "oneflow/core/kernel/foreign_input_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/foreign_blob.h"

namespace oneflow {

void ForeignInputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_input_conf().foreign_blob_buffer_name();
  std::shared_ptr<ForeignBlob> foreign_blob;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignBlob>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_blob);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  foreign_blob->CopyTo(BnInOp2Blob("out"));
}

REGISTER_KERNEL(OperatorConf::kForeignInputConf, ForeignInputKernel);

}  // namespace oneflow
