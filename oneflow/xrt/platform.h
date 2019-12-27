#ifndef ONEFLOW_XRT_PLATFORM_H_
#define ONEFLOW_XRT_PLATFORM_H_

#include "oneflow/xrt/types.h"

namespace oneflow {
namespace xrt {

namespace platform {

int GetDeviceId(const XrtDevice &device);

}  // namespace platform

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PLATFORM_H_
