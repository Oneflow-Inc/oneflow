#ifndef ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CpuDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtx);
  CpuDeviceCtx() = delete;
  ~CpuDeviceCtx() = default;

  CpuDeviceCtx(int64_t work_stream_id)
      : CpuDeviceCtx(work_stream_id, nullptr) {}
  CpuDeviceCtx(int64_t work_stream_id, CpuWorker* worker) {
    set_work_stream_id(work_stream_id);
    set_cpu_worker(worker);
  }

  void AddCallBack(std::function<void()> callback) const override {
    if (cpu_worker()) {
      cpu_worker()->PushWork(callback);
    } else {
      callback();
    }
  }

 private:
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_DEVICE_CONTEXT_H_
