#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

void ModelSaveKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  if (parallel_ctx->policy() == kDataParallel) {
    part_id_ = 0;
    part_num_ = 1;
  } else if (parallel_ctx->policy() == kModelParallel) {
    part_id_ = parallel_ctx->parallel_id();
    part_num_ = parallel_ctx->parallel_num();
  } else {
    UNEXPECTED_RUN();
  }
}

void ModelSaveKernel::ForwardDataContent(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Snapshot* snapshot = static_cast<Snapshot*>(kernel_ctx.other);
  const size_t elem_size = GetSizeOfDataType(this->kernel_conf().data_type());
  for (const std::string& ibn : kernel_conf().input_bns()) {
    const std::string& lbn = Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2Blob(ibn);
    {
      std::unique_ptr<PersistentOutStream> out_stream =
          snapshot->GetOutStream(lbn, part_id_);
      out_stream->Write(blob_ptr->dptr<char>(),
                        blob_ptr->shape().elem_cnt() * elem_size);
    }
    snapshot->OnePartDone(lbn, part_id_, part_num_);
  }
}

COMMAND(AddKernelCreator(OperatorConf::kModelSaveConf,
                         []() { return new ModelSaveKernel; }));

}  // namespace oneflow
