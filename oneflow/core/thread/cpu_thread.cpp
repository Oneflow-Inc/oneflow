#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
    PollMsgChannel(ctx);
  });
}

}  // namespace oneflow
