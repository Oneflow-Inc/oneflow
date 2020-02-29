#include "oneflow/core/kernel/categorical_hash_table_lookup_kernel_util.h"

namespace oneflow {

template<typename K, typename V>
struct CategoricalHashTableLookupKernelUtil<DeviceType::kCPU, K, V> {
  static void GetOrInsert(DeviceCtx* ctx, int64_t capacity, K* keys, V* values, V* size, int64_t n,
                          const K* hash, V* out) {
    for (int64_t i = 0; i < n; ++i) {
      const K h = hash[i];
      bool success = false;
      for (int64_t count = 0; count < capacity && !success; ++count) {
        size_t pos =
            (static_cast<size_t>(h) + static_cast<size_t>(count)) % static_cast<size_t>(capacity);
        const K k = keys[pos];
        if (k == h) {
          out[i] = values[pos];
          success = true;
          break;
        } else if (k == 0) {
          V new_size = *size + 1;
          keys[pos] = h;
          values[pos] = new_size;
          out[i] = new_size;
          *size = new_size;
          success = true;
          break;
        } else {
          continue;
        }
      }
      CHECK(success);
    }
  }
};

#define INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_CPU(key_type_pair, value_type_pair) \
  template struct CategoricalHashTableLookupKernelUtil<                                           \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(value_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_CPU,
                                 INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_CPU

}  // namespace oneflow
