#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_conf.pb.h"

namespace oneflow {

void ModelSaveKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  std::tie(part_id_, part_num_) =
      GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
}

void ModelSaveKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto tpl = static_cast<MdSaveOther*>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*tpl);
  std::get<1> (*tpl)([&](const std::string& lbn, const Blob* blob) {
    {
      std::unique_ptr<PersistentOutStream> out_stream =
          snapshot->GetOutStream(lbn, part_id_);
      out_stream->Write(blob->dptr<char>(), blob->ByteSizeOfDataContentField());
    }
    snapshot->OnePartDone(lbn, part_id_, part_num_);
  });
}

COMMAND(AddKernelCreator(OperatorConf::kModelSaveConf,
                         []() { return new ModelSaveKernel; }));

}  // namespace oneflow
