#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

template<>
void Memcpy<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz
#ifdef WITH_CUDA
                              ,
                              cudaMemcpyKind kind
#endif

) {
  if (dst == src) { return; }
  memcpy(dst, src, sz);
}

void WithHostBlobAndStreamSynchronizeEnv(DeviceCtx* ctx, Blob* blob,
                                         std::function<void(Blob*)> Callback) {
  char* host_raw_dptr = nullptr;
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->AlignedTotalByteSize()));
  Blob host_blob(MemoryCase(), &blob->blob_desc(), host_raw_dptr);
  Callback(&host_blob);
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob.dptr(), blob->ByteSizeOfBlobBody(),
                           cudaMemcpyHostToDevice);
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
  CudaCheck(cudaFreeHost(host_raw_dptr));
}

}  // namespace oneflow
