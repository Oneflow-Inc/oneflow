#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

void ModelSaveKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto save_ctx =
      static_cast<std::tuple<Snapshot*, int64_t, int64_t, ParallelPolicy>*>(
          kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  const ModelSaveKernelConf& model_save_kernel_conf =
      this->kernel_conf().model_save_conf();
  const size_t elem_size = GetSizeOfDataType(this->kernel_conf().data_type());
  int32_t part_id = model_save_kernel_conf.part_id();
  int32_t total_part_num = model_save_kernel_conf.part_num();
  for (const std::string& ibn : kernel_conf().input_bns()) {
    const std::string& lbn = Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2Blob(ibn);
    {
      std::unique_ptr<PersistentOutStream> out_stream =
          snapshot->GetOutStream(lbn, part_id);
      out_stream->Write(blob_ptr->dptr<char>(),
                        blob_ptr->shape().elem_cnt() * elem_size);
    }
    snapshot->OnePartDone(lbn, part_id, total_part_num);
  }
}

}  // namespace oneflow
