#include <pybind11/pybind11.h>
#include "oneflow/api/python/common.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
struct DeviceExportUtil final {
  static Symbol<Device> MakeDevice(const std::string& type_and_id);

  static Symbol<Device> MakeDevice(const std::string& type, int64_t device_id);
};

}
