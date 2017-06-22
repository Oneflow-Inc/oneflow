#include "oneflow\core\kernel\model_save_kernel.h"

namespace oneflow {

template<typename floating_point_type>
void ModelSaveKernel<DeviceType::kCPU, floating_point_type>::Forward(
  const KernelCtx& kernel_ctx,
  std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  std::tuple<Snapshot*, uint64_t>* save_ctx =
      static_cast<std::tuple<Snapshot*, uint64_t>>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  uint64_t parallel_id = std::get<1>(*save_ctx);
  size_t elem_size = sizeof(double);
  if(JobDesc::Singleton().floating_point_type() == FloatingPointType::kFloat) {
    elem_size = sizeof(float);
  }
  for(const std::string& ibn : op()->input_bns()) {
    std::string lbn = op()->ibn2lbn(ibn);
    PersistentOutStream* out_stream = snapshot->GetOutStream(lbn, parallel_id);
    Blob* blob_ptr = BnInOp2BlobPtr(ibn);
    kernel_ctx.device_ctx->cpu_stream()->Send([out_stream, blob_ptr]() {
      out_stream->Write(blob_ptr->dptr(),
                        blob_ptr->shape().elem_cnt() * elem_size);
    });
  }
}

}  // namespace oneflow
