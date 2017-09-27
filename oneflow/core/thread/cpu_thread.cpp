#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/device/sync_cpu_stream.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_loc_id) {
  set_thrd_loc_id(thrd_loc_id);
  cpu_device_.reset(
      new CpuDevice(JobDesc::Singleton()->use_async_cpu_stream()));
  mut_actor_thread() = std::thread([this]() {
    ThreadCtx ctx;
    ctx.cpu_stream = cpu_device_->cpu_stream();
    PollMsgChannel(ctx);
  });
}

CpuThread::~CpuThread() {
  Thread::Deconstruct();
  cpu_device_.reset();
}

}  // namespace oneflow
