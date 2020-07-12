#ifndef ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_
#define ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_

#include "oneflow/core/common/to_string.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

Maybe<const char*> DeviceTag4DeviceType(DeviceType device_type);
Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag);

template<>
inline std::string ToString(const DataType& data_type) {
  return DataType_Name(data_type);
}

template<>
inline std::string ToString(const DeviceType& device_type) {
  return CHECK_JUST(DeviceTag4DeviceType(device_type));
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_
