#ifndef ONEFLOW_CORE_DEVICE_ASYNC_CPU_STREAM_H_
#define ONEFLOW_CORE_DEVICE_ASYNC_CPU_STREAM_H_

#include "oneflow/core/device/cpu_stream.h"

namespace oneflow {

class AsyncCpuStream final : public CpuStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AsyncCpuStream);
  AsyncCpuStream() = default;
  ~AsyncCpuStream() = default;

  void SendWork(std::function<void()> work) override {
    CHECK_EQ(work_channel_.Send(work), 0);
  }

  //  0: success
  // -1: fail
  int ReceiveWork(std::function<void()>* work) override {
    return work_channel_.Receive(work);
  }

  void CloseSendEnd() override { work_channel_.CloseSendEnd(); }
  void CloseReceiveEnd() override { work_channel_.CloseReceiveEnd(); }

 private:
  Channel<std::function<void()>> work_channel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_ASYNC_CPU_STREAM_H_
