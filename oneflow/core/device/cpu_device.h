#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_H_

#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/device/sync_cpu_stream.h"

namespace oneflow {

class CpuDevice final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDevice);
  CpuDevice(bool is_async);
  ~CpuDevice();

  CpuStream* cpu_stream() { return cpu_stream_; };

 private:
  CpuStream* cpu_stream_;
  std::thread* thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_H_
