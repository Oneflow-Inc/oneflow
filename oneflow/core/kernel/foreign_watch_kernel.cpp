#include "oneflow/core/kernel/foreign_watch_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_watcher.h"

namespace oneflow {

template<>
void ForeignWatchKernel<DeviceType::kCPU>::WithInBlob(DeviceCtx* ctx, Blob* blob,
                                                      std::function<void(Blob*)> Handler) const {
  Handler(blob);
}

template<>
void ForeignWatchKernel<DeviceType::kGPU>::WithInBlob(DeviceCtx* ctx, Blob* blob,
                                                      std::function<void(Blob*)> Handler) const {
  char* host_raw_dptr = nullptr;
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->AlignedTotalByteSize()));
  MemoryCase mem_case;
  mem_case.mutable_host_mem();
  Blob host_blob(mem_case, &blob->blob_desc(), host_raw_dptr);
  Memcpy<DeviceType::kGPU>(ctx, host_blob.mut_dptr(), blob->dptr(), blob->ByteSizeOfBlobBody(),
                           cudaMemcpyDeviceToHost);
  Handler(&host_blob);
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
  CudaCheck(cudaFreeHost(host_raw_dptr));
}

template<DeviceType device_type>
void ForeignWatchKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  WithInBlob(ctx.device_ctx, BnInOp2Blob("in"), [&](Blob* in_blob) {
    OfBlob of_blob(ctx.device_ctx, in_blob);
    Global<ForeignWatcher>::Get()->Call(this->op_conf().foreign_watch_conf().handler_uuid(),
                                        reinterpret_cast<int64_t>(&of_blob));
  });
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kForeignWatchConf, DeviceType::kCPU,
                            ForeignWatchKernel<DeviceType::kCPU>);

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kForeignWatchConf, DeviceType::kGPU,
                            ForeignWatchKernel<DeviceType::kGPU>);

}  // namespace oneflow
