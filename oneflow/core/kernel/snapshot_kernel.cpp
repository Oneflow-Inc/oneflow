#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

class SnapshotKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotKernel);
  SnapshotKernel() = default;
  ~SnapshotKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<int64_t> next_snapshot_id_;
};

void SnapshotKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  CHECK_EQ(parallel_ctx->parallel_num(), 1);
  next_snapshot_id_.reset(new int64_t(1));
}

void SnapshotKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Snapshot* snapshot = Global<SnapshotMgr>::Get()->GetWriteableSnapshot(*next_snapshot_id_);
  FOR_RANGE(int64_t, i, 0, op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    const LogicalBlobId lbi = GenLogicalBlobId(op_conf().snapshot_conf().lbn(i));
    snapshot->GetOutStream(lbi)->Write(in_i->dptr<char>(), in_i->ByteSizeOfDataContentField());
  }
  snapshot->Done();
  (*next_snapshot_id_) += 1;
}

REGISTER_KERNEL(OperatorConf::kSnapshotConf, SnapshotKernel);

}  // namespace oneflow
