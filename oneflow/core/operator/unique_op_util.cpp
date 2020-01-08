#include "oneflow/core/operator/unique_op_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename U>
void GetUniqueWorkspaceSizeInBytes(int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<device_type, T, U>::GetWorkspaceSizeInBytes(nullptr, n, workspace_size_in_bytes);
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, U) func_name<device_type, T, U>
  DEFINE_STATIC_SWITCH_FUNC(void, GetUniqueWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

void UniqueOpUtil::GetWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                           DataType index_type, int64_t n,
                                           int64_t* workspace_size_in_bytes) {
  SwitchUtil::SwitchGetUniqueWorkspaceSizeInBytes(SwitchCase(device_type, value_type, index_type),
                                                  n, workspace_size_in_bytes);
}

}  // namespace oneflow
