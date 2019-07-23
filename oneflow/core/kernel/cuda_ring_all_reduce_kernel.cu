#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <device_launch_parameters.h>

namespace oneflow {

constexpr int32_t PACK_SIZE = sizeof(ulong2);
constexpr int32_t PACK_ALIGN = alignof(ulong2);
constexpr int32_t NUM_WARP = 8;
constexpr int32_t NUM_THREAD_PER_WARP = 32;
constexpr int32_t NUM_THREAD = NUM_THREAD_PER_WARP * NUM_WARP;
constexpr int32_t NUM_LINE_PER_CHUNK = 32;
constexpr int32_t NUM_PACK_PER_LINE_PER_THREAD = 8;
constexpr int32_t NUM_PACK_PER_LINE_PER_WARP = NUM_PACK_PER_LINE_PER_THREAD * NUM_THREAD_PER_WARP;
constexpr int32_t NUM_PACK_PER_LINE = NUM_PACK_PER_LINE_PER_WARP * NUM_WARP;
constexpr int32_t LINE_SIZE = NUM_PACK_PER_LINE * PACK_SIZE;
constexpr int32_t CHUNK_SIZE = LINE_SIZE * NUM_LINE_PER_CHUNK;

namespace {

template<typename T>
__global__ void AllReduceGpu(CudaRingAllReduceArg<T> arg) {}

template<typename T>
__global__ void SendGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  T* send = arg.send[block_id];
  const T* src = arg.src[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* dst_pack_ptr = reinterpret_cast<ulong2*>(send) + offset;
  const ulong2* src_pack_ptr = reinterpret_cast<const ulong2*>(src) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      line[p] = *(src_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      *(dst_pack_ptr + p * NUM_THREAD_PER_WARP) = line[p];
    }
    dst_pack_ptr += NUM_PACK_PER_LINE;
    src_pack_ptr += NUM_PACK_PER_LINE;
  }
}

}  // namespace

template<typename T>
void CudaRingAllReduceKernelUtil<T>::AllReduce(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  AllReduceGpu<<<arg.num_rings, 256, 0, ctx->cuda_stream()>>>(arg);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::Send(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  SendGpu<<<arg.num_rings, 256, 0, ctx->cuda_stream()>>>(arg);
}

#define INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct CudaRingAllReduceKernelUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
