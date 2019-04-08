#include "oneflow/core/kernel/model_save_v2_kernel.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

void ModelSaveV2Kernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  CHECK_EQ(parallel_ctx->parallel_num(), 1);
  next_snapshot_id_.reset(new int64_t(1));
}

void ModelSaveV2Kernel::Forward(const KernelCtx& ctx,
                                std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Snapshot* snapshot = Global<SnapshotMgr>::Get()->GetWriteableSnapshot(*next_snapshot_id_);
  const Blob* in = BnInOp2Blob("in");
  const LogicalBlobId lbi = GenLogicalBlobId(op_conf().model_save_v2_conf().lbn());
  snapshot->GetOutStream(lbi, 0)->Write(in->dptr<char>(), in->ByteSizeOfDataContentField());
  snapshot->OnePartDone(lbi, 0, 1);
  (*next_snapshot_id_) += 1;
}

REGISTER_KERNEL(OperatorConf::kModelSaveV2Conf, ModelSaveV2Kernel);

}  // namespace oneflow
