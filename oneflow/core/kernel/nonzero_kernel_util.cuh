#ifndef ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
#define ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T, typename OutputIter>
cudaError_t CubSelectFlagged(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                             const T* flags, OutputIter out, int32_t* num_selected);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
