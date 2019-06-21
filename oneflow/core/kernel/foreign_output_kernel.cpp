#include "oneflow/core/kernel/foreign_output_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/register/foreign_blob.h"

namespace oneflow {

void ForeignOutputKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& buffer_name = op_conf().foreign_output_conf().foreign_blob_buffer_name();
  std::shared_ptr<ForeignBlob> foreign_blob;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignBlob>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_blob);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  foreign_blob->CopyFrom(BnInOp2Blob("in"));
}

REGISTER_KERNEL(OperatorConf::kForeignOutputConf, ForeignOutputKernel);

}  // namespace oneflow
