#include "oneflow/core/framework/device_register_cpu.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {
void CpuDumpVersionInfo() {}
REGISTER_DEVICE(DeviceType::kCPU).SetDumpVersionInfoFn(CpuDumpVersionInfo).SetDeviceTag("cpu");
}
