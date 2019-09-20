#include "oneflow/core/kernel/foreign_watch_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_watcher.h"

namespace oneflow {

void ForeignWatchKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& ibns = this->op_attribute().input_bns();
  std::vector<std::unique_ptr<OfBlob>> of_blobs(ibns.size());
  Int64List of_blobs_ids;
  FOR_RANGE(int32_t, i, 0, ibns.size()) {
    of_blobs[i].reset(new OfBlob(ctx.device_ctx, BnInOp2Blob(ibns.Get(i))));
    of_blobs_ids.add_value(reinterpret_cast<int64_t>(of_blobs[i].get()));
  }
  Global<ForeignWatcher>::Get()->Call(op_conf().foreign_watch_conf().handler_uuid(),
                                      PbMessage2TxtString(of_blobs_ids));
}

REGISTER_KERNEL(OperatorConf::kForeignWatchConf, ForeignWatchKernel);

}  // namespace oneflow
