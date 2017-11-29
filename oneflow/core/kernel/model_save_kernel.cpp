#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

void ModelSaveKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  std::tie(part_id_, part_num_) =
      GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
}

void ModelSaveKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Snapshot* snapshot = static_cast<Snapshot*>(kernel_ctx.other);
  for (const std::string& ibn : kernel_conf().input_bns()) {
    const std::string& lbn = Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2Blob(ibn);
    {
      std::unique_ptr<PersistentOutStream> out_stream =
          snapshot->GetOutStream(lbn, part_id_);
      out_stream->Write(blob_ptr->dptr<char>(),
                        blob_ptr->ByteSizeOfDataContentField());
    }
    snapshot->OnePartDone(lbn, part_id_, part_num_);
  }
}

COMMAND(AddKernelCreator(OperatorConf::kModelSaveConf,
                         []() { return new ModelSaveKernel; }));

}  // namespace oneflow
