#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread() {
  mut_thread() = std::thread([this]() {
    ThreadContext ctx;
    PollMsgChannel(ctx);
  });
}

} // namespace oneflow
