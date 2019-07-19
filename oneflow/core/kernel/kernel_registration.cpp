#include "oneflow/core/kernel/kernel_registration.h"

namespace oneflow {

namespace kernel_registration {

namespace builder {

RegKeyBuilder& RegKeyBuilder::Device(DeviceType device_type) {
  CHECK(!device_) << "device_type must be set only once.";
  device_ = std::make_unique<DeviceType>(device_type);
  return *this;
}

RegKeyBuilder& RegKeyBuilder::Type(DataType dtype) {
  CHECK(!dtype_) << "data_type must be set only once.";
  dtype_ = std::make_unique<DataType>(dtype);
  return *this;
}

std::string RegKeyBuilder::Build() const {
  if (device_ && dtype_) {
    return GetHashKey(*device_, *dtype_);
  } else if (device_) {
    return GetHashKey(*device_);
  } else if (dtype_) {
    return GetHashKey(*dtype_);
  } else {
    LOG(FATAL) << "neither device_type nor data_type was set before build key";
  }
}

}  // namespace builder

}  // namespace kernel_registration

}  // namespace oneflow
