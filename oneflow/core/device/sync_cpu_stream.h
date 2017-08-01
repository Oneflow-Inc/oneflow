#ifndef ONEFLOW_CORE_DEVICE_SYNC_CPU_STREAM_H_
#define ONEFLOW_CORE_DEVICE_SYNC_CPU_STREAM_H_

#include "oneflow/core/device/cpu_stream.h"

namespace oneflow {

class SyncCpuStream final : public CpuStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncCpuStream);
  SyncCpuStream() = default;
  ~SyncCpuStream() = default;

  void SendWork(std::function<void()> work) override { work(); }

  int ReceiveWork(std::function<void()>* work) override { UNEXPECTED_RUN(); }

  void CloseSendEnd() override {}
  void CloseReceiveEnd() override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_SYNC_CPU_STREAM_H_
