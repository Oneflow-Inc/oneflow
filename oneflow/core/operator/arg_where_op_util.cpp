/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/arg_where_op_util.h"
#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename I, size_t NDims>
void GetArgWhereWorkspaceSizeInBytes(int64_t n, int64_t* workspace_bytes) {
  *workspace_bytes = static_cast<int64_t>(
      ArgWhereKernelUtil<device_type, T, I, NDims>::GetArgWhereWorkspaceSizeInBytes(nullptr, n));
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, I, N) func_name<device_type, T, I, N>
  DEFINE_STATIC_SWITCH_FUNC(void, GetArgWhereWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ),
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

void InferArgWhereWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                       DataType index_type, int32_t num_axes, int64_t n,
                                       int64_t* workspace_bytes) {
  SwitchUtil::SwitchGetArgWhereWorkspaceSizeInBytes(
      SwitchCase(device_type, value_type, index_type, num_axes), n, workspace_bytes);
}

}  // namespace oneflow
