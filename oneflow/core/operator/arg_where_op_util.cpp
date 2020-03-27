#include "oneflow/core/operator/arg_where_op_util.h"
#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename I>
void GetArgWhereWorkspaceSizeInBytes(int64_t n, int64_t* workspace_bytes) {
  *workspace_bytes = static_cast<int64_t>(ArgWhereWorkspace<device_type, T, I>()(nullptr, n));
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, I) func_name<device_type, T, I>
  DEFINE_STATIC_SWITCH_FUNC(void, GetArgWhereWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

void InferArgWhereWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                       DataType index_type, int64_t n, int64_t* workspace_bytes) {
  SwitchUtil::SwitchGetArgWhereWorkspaceSizeInBytes(SwitchCase(device_type, value_type, index_type),
                                                    n, workspace_bytes);
}

}  // namespace oneflow
