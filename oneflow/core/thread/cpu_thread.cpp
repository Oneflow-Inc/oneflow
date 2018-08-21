#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
#ifdef WITH_CUDA
    ctx.cb_event_chan = nullptr;
#endif
    PollMsgChannel(ctx);
  });
}

}  // namespace oneflow
