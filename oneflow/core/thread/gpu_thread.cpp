/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

#ifdef WITH_CUDA

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id, thrd_id]() {
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(dev_id) + " Actor : ("
                                      + std::to_string(thrd_id) + ")");
    OF_CUDA_CHECK(cudaSetDevice(dev_id));
    ThreadCtx ctx;
    ctx.g_cuda_stream.reset(new CudaStreamHandle(&cb_event_chan_));
    ctx.cb_event_chan = &cb_event_chan_;
    PollMsgChannel(ctx);
  });
  cb_event_poller_ = std::thread([this, dev_id, thrd_id]() {
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(dev_id) + " Poller : ("
                                      + std::to_string(thrd_id) + ")");
    OF_CUDA_CHECK(cudaSetDevice(dev_id));
    CudaCBEvent cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      OF_CUDA_CHECK(cudaEventSynchronize(cb_event.event));
      cb_event.callback();
      OF_CUDA_CHECK(cudaEventDestroy(cb_event.event));
    }
  });
}

GpuThread::~GpuThread() {
  cb_event_chan_.Close();
  cb_event_poller_.join();
}

REGISTER_DEVICE_THREAD_CREATOR_WITH_STREAM_ID(
    DeviceType::kGPU, ([](const StreamId& stream_id) -> Thread* {
      int64_t thrd_id = SerializeStreamIdToInt64(stream_id);
      int64_t dev_id = static_cast<int64_t>(stream_id.device_id().device_index());
      return new GpuThread(thrd_id, dev_id);
    }));

#endif

}  // namespace oneflow
