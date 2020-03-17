#ifndef ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_

#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<typename T, typename I, typename Iter>
cudaError_t CubSelectFlagged(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                             const T* flags, Iter out_iter, int32_t* num_selected);

template<typename T, typename I>
cudaError_t InferCubSelectFlaggedTempStorageBytes(DeviceCtx* ctx, int num_items, size_t& tmp_bytes);

#define ARGWHERE_SUPPORTED_DATA_TYPE_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define ARGWHERE_SUPPORTED_INDEX_TYPE_SEQ         \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_
