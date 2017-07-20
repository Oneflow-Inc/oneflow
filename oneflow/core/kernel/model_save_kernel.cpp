#include "oneflow/core/kernel/model_save_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void ModelSaveKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto save_ctx =
      static_cast<std::tuple<Snapshot*, int64_t>*>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  int64_t parallel_id = std::get<1>(*save_ctx);
  for (const std::string& ibn : op()->input_bns()) {
    const std::string& lbn = op()->Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2BlobPtr(ibn);
    kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
      std::unique_ptr<PersistentOutStream> out_stream =
          snapshot->GetOutStream(lbn, parallel_id);
      out_stream->Write(
          blob_ptr->dptr<char>(),
          blob_ptr->shape().elem_cnt() * sizeof(FloatingPointType));
    });
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(ModelSaveKernel);
REGISTER_CPU_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
