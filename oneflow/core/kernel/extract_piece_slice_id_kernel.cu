#include "oneflow/core/kernel/extract_piece_slice_id_kernel.h"

namespace oneflow {

namespace {

__global__ void ForwardOneInOutPairGpu(const int32_t instance_num, const int32_t slice_idx,
                                       int32_t* out_i_ptr) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) { out_i_ptr[i] = slice_idx; }
}

}  // namespace

void ExtractPieceSliceIdUtil<DeviceType::kGPU>::ForwardOneInOutPair(DeviceCtx* ctx,
                                                                    const int32_t instance_num,
                                                                    const int32_t slice_idx,
                                                                    int32_t* out_i_ptr) {
  ForwardOneInOutPairGpu<<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
                           ctx->cuda_stream()>>>(instance_num, slice_idx, out_i_ptr);
}

}  // namespace oneflow
