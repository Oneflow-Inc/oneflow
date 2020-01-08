#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<typename KEY, typename IDX>
struct UniqueKernelUtil<DeviceType::kCPU, KEY, IDX> {
  static void Unique(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
  static void GetWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU(key_type_pair, idx_type_pair)              \
  template struct UniqueKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(key_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU, INDEX_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU

}  // namespace oneflow
