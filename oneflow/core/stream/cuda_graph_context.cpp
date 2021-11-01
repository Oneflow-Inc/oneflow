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
#include "oneflow/core/stream/cuda_graph_context.h"

#ifdef WITH_CUDA_GRAPHS

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

CudaGraphExecutable::CudaGraphExecutable() : graph_exec_(nullptr), dev_(-1) {}

CudaGraphExecutable::~CudaGraphExecutable() { Reset(); }

void CudaGraphExecutable::Update(cudaGraph_t graph) {
  int dev = -1;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev != dev_) { Reset(); }
  dev_ = dev;
  if (graph_exec_ != nullptr) {
    cudaGraphExecUpdateResult update_result{};
    cudaGraphNode_t error_node = nullptr;
    OF_CUDA_CHECK(cudaGraphExecUpdate(graph_exec_, graph, &error_node, &update_result));
    if (update_result == cudaGraphExecUpdateSuccess) { return; }
  }
  Reset();
  OF_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, NULL, NULL, 0));
}

void CudaGraphExecutable::Launch(cudaStream_t stream) const {
  OF_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
}

bool CudaGraphExecutable::IsInstantiated() const { return graph_exec_ != nullptr; }

void CudaGraphExecutable::Reset() {
  if (graph_exec_ != nullptr) {
    CudaCurrentDeviceGuard guard(dev_);
    OF_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
}

GenericCudaGraphContext::GenericCudaGraphContext(cudaStream_t stream)
    : stream_(stream), is_graph_capturing_(false) {}

GenericCudaGraphContext::~GenericCudaGraphContext() {}

void GenericCudaGraphContext::BeginGraphCapture() {
  CHECK(!is_graph_capturing_);
  is_graph_capturing_ = true;
  OF_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
}

void GenericCudaGraphContext::EndGraphCapture(CudaGraphExecutable* executable) {
  cudaGraph_t graph = nullptr;
  OF_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  executable->Update(graph);
  OF_CUDA_CHECK(cudaGraphDestroy(graph));
  is_graph_capturing_ = false;
}

bool GenericCudaGraphContext::IsGraphCapturing() const { return is_graph_capturing_; }

void GenericCudaGraphContext::LaunchGraph(const CudaGraphExecutable* executable) {
  executable->Launch(stream_);
}

}  // namespace oneflow

#endif
