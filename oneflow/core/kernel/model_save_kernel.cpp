#include "oneflow\core\kernel\model_save_kernel.h"

namespace oneflow {

template<typename floating_point_type>
void ModelSaveKernel<DeviceType::kCPU, floating_point_type>::Forward(
  const KernelCtx& kernel_ctx,
  std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  std::tuple<Snapshot*, uint64_t> save_ctx = static_cast<std::tuple<Snapshot*, uint64_t>>(kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(save_ctx);
  uint64_t parallel_id = std::get<1>(save_ctx);
  std::vector<std::string> ibns = op()->input_bns();
  auto cpu_stream = kernel_ctx.device_ctx->cpu_stream();
  for(std::string ibn : ibns) {
    std::string lbn = op()->ibn2lbn(ibn);
    PersistentOutStream* out_stream = snapshot->GetOutStream(lbn, parallel_id);
    Blob* blob_ptr = bn_in_op2blob_ptr(ibn);
    cpu_stream->Send([&out_stream, &blob_ptr]() {
      out_stream->Write(blob_ptr->dptr(), blob_ptr->shape().elem_cnt());
    });
  }
}

}  // namespace oneflo
