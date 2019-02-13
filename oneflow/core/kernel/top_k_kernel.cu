#include "oneflow/core/kernel/top_k_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
__global__ void ForwardGpu(const T* in, const int32_t instance_num, const int32_t instance_size,
                           int32_t* out) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    T max_val = in[i * instance_size];
    int32_t max_idx = 0;
    FOR_RANGE(int32_t, j, 0, instance_size) {
      T cur_val = in[i * instance_size + j];
      if (cur_val > max_val) {
        max_val = cur_val;
        max_idx = j;
      }
    }
    out[i] = max_idx;
  }
}

template<typename T>
struct TopKKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* in, const int32_t instance_num,
                      const int32_t instance_size, const int32_t k, const bool sorted,
                      int32_t* fw_buf, int32_t* out) {
    // GPU version top_k op only support "k == 1" for now
    CHECK_EQ(k, 1);
    ForwardGpu<<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
                 ctx->cuda_stream()>>>(in, instance_num, instance_size, out);
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_TOP_K_KERNEL_UTIL

}  // namespace oneflow
