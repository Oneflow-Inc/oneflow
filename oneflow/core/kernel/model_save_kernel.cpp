#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

void ModelSaveKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  std::tie(part_id_, part_num_) = GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
}

void ModelSaveKernel::Forward(const KernelCtx& kernel_ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto tpl = static_cast<MdSaveOther*>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*tpl);
  std::get<1> (*tpl)([&](const LogicalBlobId& lbi, const Blob* blob) {
    {
      std::unique_ptr<PersistentOutStream> out_stream = snapshot->GetOutStream(lbi, part_id_);
      out_stream->Write(blob->dptr<char>(), blob->ByteSizeOfDataContentField());
    }
    snapshot->OnePartDone(lbi, part_id_, part_num_);
  });
}

REGISTER_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
