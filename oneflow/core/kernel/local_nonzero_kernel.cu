#include "oneflow/core/kernel/local_nonzero_kernel.h"

namespace oneflow {

namespace {

__device__ void SetIndex(int64_t offset, const int64_t num_elem, const int64_t* shape_ptr,
                         const int64_t shape_dim, int32_t* index_ptr) {
  int64_t dim_elem_cnt = num_elem;
  for (int64_t i = 0; i < shape_dim; ++i) {
    dim_elem_cnt /= shape_ptr[i];
    index_ptr[i] = static_cast<int32_t>(offset / dim_elem_cnt);
    offset %= dim_elem_cnt;
  }
}

template<typename T>
__global__ void LocalNonzeroForwardGpu(const int64_t num_elem, const T* in_ptr,
                                       const int64_t* shape_ptr, const int64_t shape_dim,
                                       int64_t* num_nonzero_ptr, int32_t* out_ptr) {
  int64_t num_nonzero = 0;
  for (int64_t i = 0; i < num_elem; ++i) {
    if (in_ptr[i] != ZeroVal<T>::value) {
      SetIndex(i, num_elem, shape_ptr, shape_dim, out_ptr + num_nonzero * shape_dim);
      num_nonzero += 1;
    }
  }
  *num_nonzero_ptr = num_nonzero;
}

}  // namespace

template<typename T>
void LocalNonzeroUtil<T>::ForwardGpu(DeviceCtx* ctx, const Blob* in_blob, Blob* num_nonzero_blob,
                                     Blob* shape_blob, Blob* out_blob) {
  FOR_RANGE(int64_t, i, 0, shape_blob->shape().elem_cnt()) {
    KernelUtil<DeviceType::kGPU, int64_t>::Set(ctx, in_blob->shape().At(i),
                                               shape_blob->mut_dptr<int64_t>() + i);
  }
  LocalNonzeroForwardGpu<<<1, 1, 0, ctx->cuda_stream()>>>(
      in_blob->shape().elem_cnt(), in_blob->dptr<T>(), shape_blob->dptr<int64_t>(),
      shape_blob->shape().elem_cnt(), num_nonzero_blob->mut_dptr<int64_t>(),
      out_blob->mut_dptr<int32_t>());
  CudaCheck(cudaMemcpy(out_blob->mut_dim0_valid_num_ptr(), num_nonzero_blob->dptr<int64_t>(),
                       sizeof(int64_t), cudaMemcpyDeviceToHost));
}

#define MAKE_ENTRY(type_cpp, type_proto) template struct LocalNonzeroUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
