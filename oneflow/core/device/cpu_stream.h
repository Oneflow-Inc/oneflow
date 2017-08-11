#ifndef ONEFLOW_CORE_DEVICE_CPU_STREAM_H_
#define ONEFLOW_CORE_DEVICE_CPU_STREAM_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

class CpuStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);
  CpuStream() = default;
  virtual ~CpuStream() = default;

  virtual void SendWork(std::function<void()> work) = 0;

  //  0: success
  // -1: fail
  virtual int ReceiveWork(std::function<void()>* work) = 0;

  virtual void CloseSendEnd() = 0;
  virtual void CloseReceiveEnd() = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_STREAM_H_
