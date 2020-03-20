#include "oneflow/customized/kernels/categorical_ordinal_encode_kernel_util.h"

namespace oneflow {

template<typename T>
struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCPU, T> {
  static void Encode(DeviceCtx* ctx, int64_t capacity, T* table, T* size, int64_t n, const T* hash,
                     T* out) {
    for (int64_t i = 0; i < n; ++i) {
      const T h = hash[i];
      bool success = false;
      for (int64_t count = 0; count < capacity; ++count) {
        size_t idx =
            (static_cast<size_t>(h) + static_cast<size_t>(count)) % static_cast<size_t>(capacity);
        T* k_ptr = table + idx * 2;
        T* v_ptr = k_ptr + 1;
        if (*k_ptr == h) {
          out[i] = *v_ptr;
          success = true;
          break;
        } else if (*k_ptr == 0) {
          T new_size = *size + 1;
          *k_ptr = h;
          *v_ptr = new_size;
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

#define INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU

}  // namespace oneflow
