#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_loc_id) {
  set_thrd_loc_id(thrd_loc_id);
  cpu_device_ = std::thread([this]() {
    std::function<void()> work;
    while (cpu_stream_.ReceiveWork(&work) == 0) { work(); }
  });
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
    ctx.cpu_stream = &cpu_stream_;
    PollMsgChannel(ctx);
  });
}

CpuThread::~CpuThread() {
  Thread::Deconstruct();
  cpu_stream_.CloseSendEnd();
  cpu_device_.join();
  cpu_stream_.CloseReceiveEnd();
}

}  // namespace oneflow
