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
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/device/cuda_device_descriptor.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

void SetAffinityByDevice(int64_t dev_id) {
  auto node_device_desc =
      Global<device::NodeDeviceDescriptorManager>::Get()->GetLocalNodeDeviceDescriptor();
  auto cuda_device = std::dynamic_pointer_cast<const device::CudaDeviceDescriptor>(
      node_device_desc->GetDevice(device::kCudaDeviceDescriptorClassName, dev_id));
  if (!cuda_device) { return; }
  node_device_desc->Topology()->SetCPUAffinityByPCIBusID(cuda_device->PCIBusID());
  node_device_desc->Topology()->SetMemoryAffinityByPCIBusID(cuda_device->PCIBusID());
}

}  // namespace

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id, thrd_id]() {
    SetAffinityByDevice(dev_id);
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(dev_id) + " Actor : ("
                                      + std::to_string(thrd_id) + ")");
    OF_CUDA_CHECK(cudaSetDevice(dev_id));
    ctx_.g_cuda_stream.reset(new CudaStreamHandle(&cb_event_chan_));
    // TODO(liujuncheng): force creation
    ctx_.g_cuda_stream->cuda_stream();
    ctx_.g_cuda_stream->cublas_pmh_handle();
    ctx_.g_cuda_stream->cublas_pmd_handle();
    ctx_.g_cuda_stream->cublas_tensor_op_math_handle();
    ctx_.g_cuda_stream->cudnn_handle();
    ctx_.cb_event_chan = &cb_event_chan_;
    PollMsgChannel(ctx_);
  });
  cb_event_poller_ = std::thread([this, dev_id, thrd_id]() {
    SetAffinityByDevice(dev_id);
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(dev_id) + " Poller : ("
                                      + std::to_string(thrd_id) + ")");
    OF_CUDA_CHECK(cudaSetDevice(dev_id));
    CudaCBEvent cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      ctx_.g_cuda_stream->SyncRecycleEvent(cb_event.event);
      cb_event.callback();
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
