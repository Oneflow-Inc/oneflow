#include "oneflow/core/operator/indexed_slices_reduce_sum_op_util.h"
#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void GetReduceSumWorkspaceSizeInBytes(int64_t n, int64_t m, int64_t* workspace_size_in_bytes) {
  IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::GetReduceSumWorkspaceSizeInBytes(
      nullptr, n, m, workspace_size_in_bytes);
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, GetReduceSumWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

void IndexedSlicesReduceSumOpUtil::GetReduceSumWorkspaceSizeInBytes(
    DeviceType device_type, DataType value_type, DataType index_type, int64_t n, int64_t m,
    int64_t* workspace_size_in_bytes) {
  SwitchUtil::SwitchGetReduceSumWorkspaceSizeInBytes(
      SwitchCase(device_type, value_type, index_type), n, m, workspace_size_in_bytes);
}

}  // namespace oneflow
