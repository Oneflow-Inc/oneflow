#ifndef ONEFLOW_FRAMEWORK_CORE_DEVICE_H_
#define ONEFLOW_FRAMEWORK_CORE_DEVICE_H_

#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {
namespace one {
class Device {
 public:
  Device(DeviceType device_type, int64_t device_id)
      : device_type_(device_type), device_id_(device_id) {}
  DeviceType device_type() const { return device_type_; }
  int64_t device_id() const { return device_id_; }

 private:
  DeviceType device_type_;
  int64_t device_id_;
};
}
}
#endif

