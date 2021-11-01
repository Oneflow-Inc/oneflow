#include "oneflow/serving/serving.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include <memory>
#include <utility>

using OFDevice = oneflow::Device;
using OFSymbolOfDevice = oneflow::Symbol<OFDevice>;

namespace of_serving {

struct Device::Impl {
  std::shared_ptr<OFSymbolOfDevice> device_;
  explicit Impl(const OFSymbolOfDevice& device)
      : device_(std::make_shared<OFSymbolOfDevice>(device)) {}
  explicit Impl(OFSymbolOfDevice&& device)
      : device_(std::make_shared<OFSymbolOfDevice>(std::move(device))) {}
};

namespace {

std::shared_ptr<Device> make_device(const OFSymbolOfDevice& device) {
  return std::make_shared<Device>(Device(std::make_shared<Device::Impl>(Device::Impl(device))));
}

std::shared_ptr<Device> make_device(OFSymbolOfDevice&& device) {
  return std::make_shared<Device>(
      Device(std::make_shared<Device::Impl>(Device::Impl(std::move(device)))));
}

}  // namespace

void Device::CheckDeviceType(const std::string& type) {
  if (OFDevice::type_supported.find(type) == OFDevice::type_supported.end()) {
    std::string error_msg =
        "Expected one of cpu, cuda device type at start of device string " + type;
    throw std::runtime_error(error_msg);
  }
}

std::shared_ptr<Device> Device::New(const std::string& type) {
  CheckDeviceType(type);
  return make_device(OFDevice::New(type).GetOrThrow());
}

std::shared_ptr<Device> Device::New(const std::string& type, int64_t device_id) {
  CheckDeviceType(type);
  return make_device(OFDevice::New(type, device_id).GetOrThrow());
}

std::shared_ptr<Device> Device::ParseAndNew(const std::string& type_and_id) {
  std::string type;
  int device_id = -1;
  oneflow::ParsingDeviceTag(type_and_id, &type, &device_id).GetOrThrow();
  if (device_id == -1) {
    return New(type);
  } else {
    return New(type, device_id);
  }
}

}  // namespace of_serving
