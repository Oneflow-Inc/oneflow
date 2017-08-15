#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/device/sync_cpu_stream.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_loc_id) {
  set_thrd_loc_id(thrd_loc_id);
  if (JobDesc::Singleton()->use_async_cpu_stream()) {
    cpu_stream_.reset(new AsyncCpuStream);
    cpu_device_.reset(new std::thread([this]() {
      std::function<void()> work;
      while (true) {
        int res = cpu_stream_->ReceiveWork(&work);
        if (of_likely(res == 0)) {
          work();
        } else if (of_likely(res == 1)) {
          continue;
        } else if (of_likely(res == -1)) {
          break;
        } else {
          UNEXPECTED_RUN();
        }
      }
    }));
  } else {
    cpu_stream_.reset(new SyncCpuStream);
  }
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
    ctx.cpu_stream = cpu_stream_.get();
    PollMsgChannel(ctx);
  });
}

CpuThread::~CpuThread() {
  Thread::Deconstruct();
  cpu_stream_->CloseSendEnd();
  if (cpu_device_) { cpu_device_->join(); }
  cpu_stream_->CloseReceiveEnd();
}

}  // namespace oneflow
