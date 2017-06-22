#include "oneflow/core/kernel/model_save_kernel.h"

namespace oneflow {

template<typename floating_point_type>
void ModelSaveKernel<DeviceType::kCPU, floating_point_type>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto save_ctx = static_cast<std::tuple<Snapshot*, uint64_t>*>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  uint64_t parallel_id = std::get<1>(*save_ctx);
  for(const std::string& ibn : op()->input_bns()) {
    const std::string& lbn = op()->Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2BlobPtr(ibn);
    kernel_ctx.device_ctx->cpu_stream()->Send([=]() {
      std::unique_ptr<PersistentOutStream> out_stream = 
          snapshot->GetOutStream(lbn, parallel_id);
      out_stream->Write(static_cast<const char*>(blob_ptr->dptr()),
                        blob_ptr->shape().elem_cnt() * sizeof(floating_point_type));
    });
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(ModelSaveKernel);
REGISTER_CPU_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
