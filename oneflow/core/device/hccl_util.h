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
#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#ifdef WITH_NPU
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/npu_util.h"
#include "hccl/hccl.h"


namespace oneflow {

extern std::map<enum DataType, HcclDataType> hcclDataType;

inline HcclDataType GetHcclDataType(const DataType& type) {
  try {
    return hcclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for HCCL process group");
  }
}

std::string HcclUniqueIdToString(const HcclRootInfo& unique_id);

void HcclUniqueIdFromString(const std::string& str, HcclRootInfo* unique_id);



}  // namespace oneflow
#endif  // WITH_NPU

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
