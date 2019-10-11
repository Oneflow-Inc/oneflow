#ifndef ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
#define ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T>
cudaError_t CubReduceCount(void* tmp, size_t& tmp_bytes, const T* in, int32_t* out, int num_items,
                           cudaStream_t stream);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
