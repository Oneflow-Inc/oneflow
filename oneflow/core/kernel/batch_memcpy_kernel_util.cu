#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace {

constexpr int32_t kMaxBatchSize = 128;
constexpr int32_t kMaxCopySize = 4 * 1024 * 1024;

using BulkType = int4;

struct BatchMemcpyParam {
  MemcpyParam params[kMaxBatchSize];
  int32_t batch_size;
};

__global__ void GpuCopy(BatchMemcpyParam batch_memcpy_param) {
  for (int b = blockIdx.x; b < batch_memcpy_param.batch_size; b += gridDim.x) {
    const int fast_count = batch_memcpy_param.params[b].count / sizeof(BulkType);
    BulkType* fast_dst = reinterpret_cast<BulkType*>(batch_memcpy_param.params[b].dst);
    const BulkType* fast_src = reinterpret_cast<const BulkType*>(batch_memcpy_param.params[b].src);
#pragma unroll(5)
    for (int t = threadIdx.x; t < fast_count; t += blockDim.x) { fast_dst[t] = fast_src[t]; }
    const int fast_bytes = fast_count * sizeof(BulkType);
    const int slow_count = batch_memcpy_param.params[b].count - fast_bytes;
    unsigned char* slow_dst =
        reinterpret_cast<unsigned char*>(batch_memcpy_param.params[b].dst) + fast_bytes;
    const unsigned char* slow_src =
        reinterpret_cast<const unsigned char*>(batch_memcpy_param.params[b].src) + fast_bytes;
    for (int t = threadIdx.x; t < slow_count; t += blockDim.x) { slow_dst[t] = slow_src[t]; }
  }
}

}  // namespace

template<>
void BatchMemcpyKernelUtil<DeviceType::kGPU>::Copy(DeviceCtx* ctx,
                                                   const std::vector<MemcpyParam>& params) {
  if (params.size() == 1) {
    CudaCheck(cudaMemcpyAsync(params.front().dst, params.front().src, params.front().count,
                              cudaMemcpyDefault, ctx->cuda_stream()));
  } else {
    int block_size = 0;
    int num_blocks = 0;
    CudaCheck(cudaOccupancyMaxPotentialBlockSize(&num_blocks, &block_size, GpuCopy));
    BatchMemcpyParam batch_memcpy_param{};
    batch_memcpy_param.batch_size = 0;
    for (const MemcpyParam& param : params) {
      if (reinterpret_cast<uintptr_t>(param.dst) % sizeof(BulkType) == 0
          && reinterpret_cast<uintptr_t>(param.src) % sizeof(BulkType) == 0
          && param.count <= kMaxCopySize) {
        batch_memcpy_param.params[batch_memcpy_param.batch_size] = param;
        batch_memcpy_param.batch_size += 1;
        if (batch_memcpy_param.batch_size == kMaxBatchSize) {
          GpuCopy<<<num_blocks, block_size, 0, ctx->cuda_stream()>>>(batch_memcpy_param);
          batch_memcpy_param.batch_size = 0;
        }
      } else {
        CudaCheck(cudaMemcpyAsync(param.dst, param.src, param.count, cudaMemcpyDefault,
                                  ctx->cuda_stream()));
      }
    }
    if (batch_memcpy_param.batch_size != 0) {
      GpuCopy<<<num_blocks, block_size, 0, ctx->cuda_stream()>>>(batch_memcpy_param);
    }
  }
}

}  // namespace oneflow
