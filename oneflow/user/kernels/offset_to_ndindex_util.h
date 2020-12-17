#ifndef ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_
#define ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_

#ifdef __CUDA_ARCH__
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"
#endif

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow{

#define OFFSET_TO_NDINDEX_DATA_TYPE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

const int index_max_ndims = 6; 

template<typename IDX_T>
using IndexHelper = NdIndexOffsetHelper<IDX_T, index_max_ndims>;

namespace user_op{
template<DeviceType device_type, typename T>
struct OffsetToNdIndexFunctor final {
    void operator()(DeviceCtx* ctx, int32_t in_num, 
                    int32_t ndim, const T* index, const T* dims_tensor, T* out);
};

#ifdef __CUDA_ARCH__
// TODO: Two versions of check
__device__ void checkOffsetGPU(int32_t offset, int32_t dims_elem_cnt);
#endif

inline void checkOffsetCPU(int32_t offset, int32_t dims_elem_cnt) {
  CHECK_LE(offset, dims_elem_cnt);
}

template<typename T>
OF_DEVICE_FUNC void DoOffsetToIndex(int32_t in_num, 
                    int32_t ndim, const T* index, const T* dims, T* out){
    IndexHelper<T> helper(dims, ndim);
    int offset = *index;
    int dims_elem_cnt = 1;
    for(int i=0; i < ndim; i++) { 
      dims_elem_cnt = dims_elem_cnt * dims[i]; 
    }
    // printf("Dims elem cnt is %d", dims_elem_cnt);
    
    // TODO： Add Check(zhengzekang)
#ifdef __CUDA_ARCH__
    checkOffsetGPU(offset, dims_elem_cnt);
#else
    checkOffsetCPU(offset, dims_elem_cnt);
#endif

    helper.OffsetToNdIndex(offset, out);
}

#define INSTANTIATE_OFFSET_TO_NDINDEX_FUNCTOR(device_type_v, dtype_pair) \
  template struct OffsetToNdIndexFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

} // namespace user_op
} // namespace oneflow

# endif // ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_
