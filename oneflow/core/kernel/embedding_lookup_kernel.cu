#include "oneflow/core/kernel/embedding_lookup_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void EmbeddingLookupForward(const int32_t n, const int32_t units, const int32_t* in_dptr,
                                       const T* weight_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t oidx = i / units;
    const int32_t uidx = i % units;
    const int32_t idx = in_dptr[oidx];
    if (idx == -1) { continue; }
    out_dptr[i] = weight_dptr[idx * units + uidx];
  }
}

}  // namespace

template<typename T>
class EmbeddingLookupKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupKernelUtil);
  EmbeddingLookupKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const Blob* in_blob, const Blob* weight_blob,
                      Blob* out_blob) {
    const int32_t* in_dptr = in_blob->dptr<int32_t>();
    const T* weight_dptr = weight_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();
    const int32_t units = out_blob->shape().Count(1);
    const int32_t n = out_blob->shape().elem_cnt();
    EmbeddingLookupForward<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, units, in_dptr, weight_dptr, out_dptr);
  }
};

#define INSTANTIATE_EMBEDDING_LOOKUP_KERNEL_UTIL(type_cpp, type_proto) \
  template class EmbeddingLookupKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_EMBEDDING_LOOKUP_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
