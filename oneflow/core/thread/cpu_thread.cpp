#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread() {
  cpu_device_ = std::thread([this]() {
    std::function<void()> work;
    while (cpu_stream_.Receive(&work) == 0) { work(); }
  });
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
    ctx.cpu_stream = &cpu_stream_;
    PollMsgChannel(ctx);
  });
}

CpuThread::~CpuThread() {
  cpu_stream_.CloseSendEnd();
  cpu_device_.join();
  cpu_stream_.CloseReceiveEnd();
}

}  // namespace oneflow
