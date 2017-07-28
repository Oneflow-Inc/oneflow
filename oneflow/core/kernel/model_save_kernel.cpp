#include "oneflow/core/kernel/model_save_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void ModelSaveKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto save_ctx =
      static_cast<std::tuple<Snapshot*, int64_t, int64_t, ParallelPolicy>*>(
          kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  int64_t parallel_id = std::get<1>(*save_ctx);
  int64_t parallel_num = std::get<2>(*save_ctx);
  ParallelPolicy policy = std::get<3>(*save_ctx);
  int32_t part_id = -1;
  int32_t total_part_num = -1;
  if (policy == kDataParallel) {
    part_id = 0;
    total_part_num = 1;
    CHECK_EQ(parallel_id, 0);
  } else if (policy == kModelParallel) {
    part_id = parallel_id;
    total_part_num = parallel_num;
  } else {
    UNEXPECTED_RUN();
  }
  for (const std::string& ibn : op()->input_bns()) {
    const std::string& lbn = op()->Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2BlobPtr(ibn);
    kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
      {
        std::unique_ptr<PersistentOutStream> out_stream =
            snapshot->GetOutStream(lbn, part_id, total_part_num);
        out_stream->Write(
            blob_ptr->dptr<char>(),
            blob_ptr->shape().elem_cnt() * sizeof(FloatingPointType));
      }
      snapshot->OnePartDone4Lbn(lbn);
    });
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(ModelSaveKernel);
REGISTER_CPU_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
