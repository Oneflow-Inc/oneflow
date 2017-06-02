#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread() {
  mut_thread() = std::thread([this]() {
    PollMsgChannel();
  });
}

} // namespace oneflow
