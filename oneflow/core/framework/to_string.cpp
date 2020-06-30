#include "oneflow/core/framework/to_string.h"

namespace oneflow {

Maybe<const char*> DeviceTag4DeviceType(DeviceType device_type) {
  if (device_type == kCPU) { return "cpu"; }
  if (device_type == kGPU) { return "gpu"; }
  return Error::DeviceTagNotFound() << "invalid";
}

Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag) {
  if (device_tag == "cpu") { return DeviceType::kCPU; }
  if (device_tag == "gpu") { return DeviceType::kGPU; }
  return Error::DeviceTagNotFound() << "device tag `" << device_tag << "' not found";
}

}  // namespace oneflow
