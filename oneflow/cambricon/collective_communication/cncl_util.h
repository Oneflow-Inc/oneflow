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
#ifndef ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_UTIL_H_
#define ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_UTIL_H_

#include <glog/logging.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.pb.h"

#include <cncl.h>

namespace oneflow {

#define OF_CNCL_CHECK(condition)                                                            \
  for (cnclResult_t _of_cncl_check_status = (condition);                                    \
       _of_cncl_check_status != cnclResult_t::CNCL_RET_SUCCESS;)                            \
  LOG(FATAL) << "Check failed: " #condition " : " << cnclGetErrorStr(_of_cncl_check_status) \
             << " (" << _of_cncl_check_status << "). "                                      \
             << "To see more detail, please run OneFlow with system variable CNCL_LOG_LEVEL=DEBUG"

inline cnclDataType_t GetCnclDataType(const DataType& dt) {
  switch (dt) {
#define CNCL_DATA_TYPE_CASE(dtype) \
  case DataType::k##dtype: return cncl##dtype
    CNCL_DATA_TYPE_CASE(Char);
    CNCL_DATA_TYPE_CASE(Float);
    CNCL_DATA_TYPE_CASE(Int8);
    CNCL_DATA_TYPE_CASE(Int16);
    CNCL_DATA_TYPE_CASE(Int32);
    CNCL_DATA_TYPE_CASE(Float16);
    case DataType::kBool: return cnclUint8;
    case DataType::kUInt8: return cnclUint8;
    case DataType::kUInt16: return cnclUint16;
    case DataType::kUInt32: return cnclUint32;
    case DataType::kInvalidDataType: return cnclInvalid;
    default:
      THROW(RuntimeError)
          << "No corresponding cncl dtype: " << DataType_Name(dt)
          << "! Please convert to the other supported data type of cncl: char, int8, uint8, "
             "int16, uint16, int, uint, float, float16!";
  }
#undef CNCL_DATA_TYPE_CASE
  return cnclFloat;
}

std::string CnclCliqueIdToString(const cnclCliqueId& unique_id);

void CnclCliqueIdFromString(const std::string& str, cnclCliqueId* unique_id);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_UTIL_H_
