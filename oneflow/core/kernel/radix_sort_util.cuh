#ifndef RADIX_SORT_UTIL_CUH_
#define RADIX_SORT_UTIL_CUH_

#include <cub/cub.cuh>

namespace oneflow {

template<typename KeyType, typename ValueType>
void SortPairsAscending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                        int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                        KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                        cudaStream_t cuda_stream);

template<typename KeyType, typename ValueType>
void SortPairsDescending(const KeyType* keys_ptr, const ValueType* values_ptr, int32_t num_row,
                         int32_t num_col, void* temp_storage_ptr, int32_t temp_storage_bytes,
                         KeyType* sorted_keys_ptr, ValueType* sorted_values_ptr,
                         cudaStream_t cuda_stream);

}  // namespace oneflow

#endif RADIX_SORT_UTIL_CUH_
