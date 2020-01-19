#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {

#ifdef WITH_CUDA

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id]() {
    CudaCheck(cudaSetDevice(dev_id));
    ThreadCtx ctx;
    ctx.g_cuda_stream.reset(new CudaStreamHandle(&cb_event_chan_));
    ctx.cb_event_chan = &cb_event_chan_;
    PollMsgChannel(ctx);
  });
  cb_event_poller_ = std::thread([this, dev_id]() {
    CudaCheck(cudaSetDevice(dev_id));
    CudaCBEvent cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      // const std::string mark_sync("gpu cb sync " + cb_event.op_name);
      // nvtxRangePush(mark_sync.c_str());
      CudaCheck(cudaEventSynchronize(cb_event.event));
      // nvtxRangePop();

      // const std::string mark_cb("gpu cb call " + cb_event.op_name);
      // nvtxRangePush(mark_cb.c_str());
      cb_event.callback();
      // nvtxRangePop();

      // const std::string mark_des("gpu cb destroy " + cb_event.op_name);
      // nvtxRangePush(mark_des.c_str());
      CudaCheck(cudaEventDestroy(cb_event.event));
      // nvtxRangePop();
    }
  });
}

GpuThread::~GpuThread() {
  cb_event_chan_.Close();
  cb_event_poller_.join();
}

#endif

}  // namespace oneflow
