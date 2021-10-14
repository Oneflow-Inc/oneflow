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
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/stream/cuda_graph_context.h"
#include "oneflow/core/stream/execution_context_hook.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/device/cuda_device_descriptor.h"
#include "oneflow/core/device/cuda_event_record.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cuda_check_numerics_kernel_observer.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace {

constexpr int kCudaEventReuseRecycleThreshold = 32;
constexpr size_t kDefaultWorkspaceSize = 4 * 1024 * 1024;  // 4M

void SetAffinityByDevice(int64_t dev_id) {
  auto node_device_desc =
      Global<device::NodeDeviceDescriptorManager>::Get()->GetLocalNodeDeviceDescriptor();
  auto cuda_device = std::dynamic_pointer_cast<const device::CudaDeviceDescriptor>(
      node_device_desc->GetDevice(device::kCudaDeviceDescriptorClassName, dev_id));
  if (!cuda_device) { return; }
  node_device_desc->Topology()->SetCPUAffinityByPCIBusID(cuda_device->PCIBusID());
  node_device_desc->Topology()->SetMemoryAffinityByPCIBusID(cuda_device->PCIBusID());
}

#ifdef WITH_CUDA_GRAPHS
#define CUDA_STREAM_CONTEXT_IMPL_BASE                                        \
 public                                                                      \
  CudaStreamContext, public CudaGraphContext, public KernelObserverProvider, \
      public ExecutionContextHook
#else
#define CUDA_STREAM_CONTEXT_IMPL_BASE \
 public                               \
  CudaStreamContext, public KernelObserverProvider, public ExecutionContextHook
#endif

class CudaStreamContextImpl : CUDA_STREAM_CONTEXT_IMPL_BASE {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamContextImpl);
  explicit CudaStreamContextImpl(const StreamId& stream_id);
  virtual ~CudaStreamContextImpl();

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  Maybe<void> AddCallback(std::function<void()> callback) override;
  Maybe<void> Sync() override;
  DeviceType device_type() const override { return DeviceType::kGPU; }
  KernelObserver* GetKernelObserver() override;

  cudaStream_t cuda_stream() const override;
  cublasHandle_t cublas_handle() const override;
  cudnnHandle_t cudnn_handle() const override;

#ifdef WITH_CUDA_GRAPHS
  void BeginGraphCapture() override;
  void EndGraphCapture(CudaGraphExecutable* executable) override;
  bool IsGraphCapturing() const override;
  void LaunchGraph(const CudaGraphExecutable* executable) override;
#endif

 private:
  cudaEvent_t GetEvent();
  void SyncRecycleEvent(cudaEvent_t event);

  Channel<std::pair<cudaEvent_t, std::function<void()>>> cb_event_chan_;
  cudaStream_t cuda_stream_{};
  cublasHandle_t cublas_handle_{};
  cudnnHandle_t cudnn_handle_{};
  int cuda_event_flags_;
  bool reuse_cuda_event_;
  std::deque<cudaEvent_t> consumer_event_queue_;
  std::deque<cudaEvent_t> producer_event_queue_;
  std::deque<cudaEvent_t> global_event_queue_;
  std::mutex global_event_queue_mutex_;
  std::thread poller_thread_;
  StreamId stream_id_;
  std::unique_ptr<KernelObserver> kernel_observer_;
#ifdef WITH_CUDA_GRAPHS
  std::unique_ptr<GenericCudaGraphContext> cuda_graph_ctx_impl_;
#endif  // WITH_CUDA_GRAPHS
#if CUDA_VERSION >= 11000
  void* workspace_{};
  size_t workspace_size_{};
#endif  // CUDA_VERSION >= 11000
};

}  // namespace

CudaStreamContextImpl::CudaStreamContextImpl(const StreamId& stream_id) : stream_id_(stream_id) {
  CudaCurrentDeviceGuard guard(stream_id_.device_id().device_index());
  CHECK_EQ(stream_id.device_id().device_type(), DeviceType::kGPU);
  cuda_event_flags_ = cudaEventDisableTiming;
  if (ParseBooleanFromEnv("ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC", false)) {
    cuda_event_flags_ |= cudaEventBlockingSync;
  }
  reuse_cuda_event_ = ParseBooleanFromEnv("ONEFLOW_STREAM_REUSE_CUDA_EVENT", false);

  // cuda_stream
  OF_CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

  // cublas_handle
  OF_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  OF_CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_));
#if CUDA_VERSION >= 11000
  if (Global<ResourceDesc, ForSession>::Get()->enable_tensor_float_32_compute()) {
    OF_CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
  }
  workspace_size_ = kDefaultWorkspaceSize;
  OF_CUDA_CHECK(cudaMalloc(&workspace_, workspace_size_));
  OF_CUBLAS_CHECK(cublasSetWorkspace(cublas_handle_, workspace_, workspace_size_));
#endif

  // cudnn_handle
  if (IsCuda9OnTuringDevice()) {
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaGetLastError());
  }
  OF_CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
  if (IsCuda9OnTuringDevice()) {
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    cudaGetLastError();
  }
  OF_CUDNN_CHECK(cudnnSetStream(cudnn_handle_, cuda_stream_));

  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    LOG(WARNING) << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS has been set "
                    "to a truthy "
                    "value, it will impact performance";
    kernel_observers.emplace_back(new CudaCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));

#ifdef WITH_CUDA_GRAPHS
  cuda_graph_ctx_impl_.reset(new GenericCudaGraphContext(cuda_stream_));
#endif  // WITH_CUDA_GRAPHS

  poller_thread_ = std::thread([this, stream_id]() {
    int dev_id = stream_id.device_id().device_index();
    CudaCurrentDeviceGuard guard(dev_id);
    SetAffinityByDevice(dev_id);
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(dev_id) + " Poller : ("
                                      + std::to_string(stream_id.stream_index()) + ")");
    std::pair<cudaEvent_t, std::function<void()>> cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      SyncRecycleEvent(cb_event.first);
      cb_event.second();
    }
  });
}

CudaStreamContextImpl::~CudaStreamContextImpl() {
  CudaCurrentDeviceGuard guard(stream_id_.device_id().device_index());
  cb_event_chan_.Close();
  poller_thread_.join();
  OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  for (cudaEvent_t event : consumer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : producer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : global_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  OF_CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
  OF_CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
#if CUDA_VERSION >= 11000
  OF_CUDA_CHECK(cudaFree(workspace_));
#endif  // CUDA_VERSION >= 11000
}

Maybe<void> CudaStreamContextImpl::OnExecutionContextSetup() {
  SetAffinityByDevice(stream_id_.device_id().device_index());
  OF_CUDA_CHECK(cudaSetDevice(stream_id_.device_id().device_index()));
  return Maybe<void>::Ok();
}

Maybe<void> CudaStreamContextImpl::OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

Maybe<void> CudaStreamContextImpl::AddCallback(std::function<void()> callback) {
  cudaEvent_t cuda_event = GetEvent();
  OF_CUDA_CHECK(cudaEventRecord(cuda_event, cuda_stream_));
  cb_event_chan_.Send(std::make_pair(cuda_event, std::move(callback)));
  return Maybe<void>::Ok();
}

cudaEvent_t CudaStreamContextImpl::GetEvent() {
  cudaEvent_t cuda_event{};
  if (reuse_cuda_event_) {
    if (consumer_event_queue_.empty()) {
      std::unique_lock<std::mutex> lock(global_event_queue_mutex_);
      consumer_event_queue_.swap(global_event_queue_);
    }
    if (consumer_event_queue_.empty()) {
      OF_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cuda_event_flags_));
    } else {
      cuda_event = consumer_event_queue_.back();
      consumer_event_queue_.pop_back();
    }
  } else {
    OF_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cuda_event_flags_));
  }
  return cuda_event;
}

void CudaStreamContextImpl::SyncRecycleEvent(cudaEvent_t event) {
  OF_CUDA_CHECK(cudaEventSynchronize(event));
  if (reuse_cuda_event_) {
    producer_event_queue_.push_back(event);
    if (producer_event_queue_.size() >= kCudaEventReuseRecycleThreshold) {
      std::unique_lock<std::mutex> lock(global_event_queue_mutex_);
      global_event_queue_.insert(global_event_queue_.end(), producer_event_queue_.begin(),
                                 producer_event_queue_.end());
      producer_event_queue_.clear();
    }
  } else {
    OF_CUDA_CHECK(cudaEventDestroy(event));
  }
}

Maybe<void> CudaStreamContextImpl::Sync() {
  cudaError_t err = cudaStreamSynchronize(cuda_stream_);
  if (err == cudaSuccess) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << cudaGetErrorString(err) << " (" << err << ") ";
  }
}

KernelObserver* CudaStreamContextImpl::GetKernelObserver() { return kernel_observer_.get(); }

cudaStream_t CudaStreamContextImpl::cuda_stream() const { return cuda_stream_; }

cublasHandle_t CudaStreamContextImpl::cublas_handle() const { return cublas_handle_; }

cudnnHandle_t CudaStreamContextImpl::cudnn_handle() const { return cudnn_handle_; }

#ifdef WITH_CUDA_GRAPHS

void CudaStreamContextImpl::BeginGraphCapture() {
  return cuda_graph_ctx_impl_->BeginGraphCapture();
}

void CudaStreamContextImpl::EndGraphCapture(CudaGraphExecutable* executable) {
  return cuda_graph_ctx_impl_->EndGraphCapture(executable);
}

bool CudaStreamContextImpl::IsGraphCapturing() const {
  return cuda_graph_ctx_impl_->IsGraphCapturing();
}

void CudaStreamContextImpl::LaunchGraph(const CudaGraphExecutable* executable) {
  return cuda_graph_ctx_impl_->LaunchGraph(executable);
}

#endif

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(DeviceType::kGPU,
                                               ([](const StreamId& stream_id) -> StreamContext* {
                                                 return new CudaStreamContextImpl(stream_id);
                                               }));

}  // namespace oneflow

#endif
