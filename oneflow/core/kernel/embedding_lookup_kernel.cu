#include "oneflow/core/kernel/embedding_lookup_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void EmbeddingLookupBackward(const int32_t n, const int32_t* in_dptr,
                                        const T* out_diff_dptr,
                                        const int32_t units,
                                        T* weight_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int32_t oidx = i / units;
    const int32_t uidx = i % units;
    const int32_t idx = in_dptr[oidx];
    gpu_atomic_add<T>(weight_diff_dptr + idx * units + uidx,
                      out_diff_dptr[oidx * units + uidx]);
  }
}

}  // namespace

template<typename T>
class EmbeddingLookupKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupKernelUtil);
  EmbeddingLookupKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const Blob* in_blob,
                      const Blob* weight_blob, Blob* out_blob) {
    const int32_t* in_dptr = in_blob->dptr<int32_t>();
    const T* weight_dptr = weight_blob->dptr<T>();
    const int32_t units = out_blob->shape().Count(1);
    T* out_dptr = out_blob->mut_dptr<T>();

    FOR_RANGE(int32_t, n, 0, in_blob->shape().At(0)) {
      CHECK(in_dptr[n] < weight_blob->shape().At(0));
      const int32_t idx = in_dptr[n];
      Memcpy<DeviceType::kGPU>(ctx, out_dptr + n * units,
                               weight_dptr + idx * units, units);
    }
  }

  static void Backward(DeviceCtx* ctx, const Blob* in_blob,
                       const Blob* out_diff_blob, Blob* weight_diff_blob) {
    const int32_t count = out_diff_blob->shape().elem_cnt();
    EmbeddingLookupBackward<<<BlocksNum4ThreadsNum(count),
                              kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        count, in_blob->dptr<int32_t>(), out_diff_blob->dptr<T>(),
        out_diff_blob->shape().Count(1), weight_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_EMBEDDING_LOOKUP_KERNEL_UTIL(type_cpp, type_proto) \
  template class EmbeddingLookupKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_EMBEDDING_LOOKUP_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
