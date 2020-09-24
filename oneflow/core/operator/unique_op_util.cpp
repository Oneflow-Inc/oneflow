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
#include "oneflow/core/operator/unique_op_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename U>
void GetUniqueWorkspaceSizeInBytes(int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<device_type, T, U>::GetUniqueWorkspaceSizeInBytes(nullptr, n,
                                                                     workspace_size_in_bytes);
}

template<DeviceType device_type, typename T, typename U>
void GetUniqueWithCountsWorkspaceSizeInBytes(int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<device_type, T, U>::GetUniqueWithCountsWorkspaceSizeInBytes(
      nullptr, n, workspace_size_in_bytes);
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, U) func_name<device_type, T, U>
  DEFINE_STATIC_SWITCH_FUNC(void, GetUniqueWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
  DEFINE_STATIC_SWITCH_FUNC(void, GetUniqueWithCountsWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

void UniqueOpUtil::GetUniqueWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                                 DataType index_type, int64_t n,
                                                 int64_t* workspace_size_in_bytes) {
  SwitchUtil::SwitchGetUniqueWorkspaceSizeInBytes(SwitchCase(device_type, value_type, index_type),
                                                  n, workspace_size_in_bytes);
}

void UniqueOpUtil::GetUniqueWithCountsWorkspaceSizeInBytes(DeviceType device_type,
                                                           DataType value_type, DataType index_type,
                                                           int64_t n,
                                                           int64_t* workspace_size_in_bytes) {
  SwitchUtil::SwitchGetUniqueWithCountsWorkspaceSizeInBytes(
      SwitchCase(device_type, value_type, index_type), n, workspace_size_in_bytes);
}

}  // namespace oneflow
