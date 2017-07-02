#ifndef ONEFLOW_CORE_DEVICE_CPU_STREAM_H_
#define ONEFLOW_CORE_DEVICE_CPU_STREAM_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

class CpuStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);
  CpuStream() = default;
  ~CpuStream() = default;

  void SendWork(std::function<void()> work) {
    CHECK_EQ(work_channel_.Send(work), 0);
  }

  //  0: success
  // -1: fail
  int ReceiveWork(std::function<void()>* work) {
    return work_channel_.Receive(work);
  }

  void CloseSendEnd() { work_channel_.CloseSendEnd(); }
  void CloseReceiveEnd() { work_channel_.CloseReceiveEnd(); }

 private:
  Channel<std::function<void()>> work_channel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_STREAM_H_
