#ifndef ONEFLOW_XRT_PLATFORM_H_
#define ONEFLOW_XRT_PLATFORM_H_

#include "oneflow/xrt/types.h"

namespace oneflow {
namespace xrt {

namespace platform {

int GetDeviceId(const XrtDevice &device);

void SetDeviceId(const XrtDevice &device, const int device_id);

}  // namespace platform

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PLATFORM_H_
