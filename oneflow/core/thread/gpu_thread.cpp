#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include <pthread.h>
#include <sched.h>

namespace oneflow {

namespace {

void SetRealtimePriority() {
  struct sched_param params {};
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  pthread_t this_thread = pthread_self();
  if (pthread_setschedparam(this_thread, SCHED_FIFO, &params) != 0) {
    LOG(WARNING) << "err set realtime proiority";
  }
}

}  // namespace

#ifdef WITH_CUDA

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id]() {
    SetRealtimePriority();
    CudaCheck(cudaSetDevice(dev_id));
    ThreadCtx ctx;
    ctx.g_cuda_stream.reset(new CudaStreamHandle(&cb_event_chan_));
    ctx.cb_event_chan = &cb_event_chan_;
    PollMsgChannel(ctx);
  });
  cb_event_poller_ = std::thread([this, dev_id]() {
    SetRealtimePriority();
    CudaCheck(cudaSetDevice(dev_id));
    CudaCBEvent cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      CudaCheck(cudaEventSynchronize(cb_event.event));
      cb_event.callback();
      CudaCheck(cudaEventDestroy(cb_event.event));
    }
  });
}

GpuThread::~GpuThread() {
  cb_event_chan_.Close();
  cb_event_poller_.join();
}

#endif

}  // namespace oneflow
