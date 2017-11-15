#include "oneflow/core/device/cpu_device.h"

namespace oneflow {

CpuDevice::CpuDevice(bool is_async) {
  if (is_async) {
    cpu_stream_ = new AsyncCpuStream;
    thread_ = new std::thread([this]() {
      std::function<void()> work;
      while (cpu_stream_->ReceiveWork(&work) == 0) { work(); }
    });
  } else {
    cpu_stream_ = new SyncCpuStream;
    thread_ = nullptr;
  }
}

CpuDevice::~CpuDevice() {
  cpu_stream_->CloseSendEnd();
  if (thread_) { thread_->join(); }
  cpu_stream_->CloseReceiveEnd();
  delete cpu_stream_;
  if (thread_) { delete thread_; }
}

}  // namespace oneflow
