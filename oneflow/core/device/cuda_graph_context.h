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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_GRAPH_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_GRAPH_CONTEXT_H_

#include "oneflow/core/common/util.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 11000
#define WITH_CUDA_GRAPHS

namespace oneflow {

class CudaGraphContext {
 public:
  CudaGraphContext(cudaStream_t stream) : stream_(stream), graph_exec_(nullptr) {}
  ~CudaGraphContext() {
    if (graph_exec_ != nullptr) { OF_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_)); }
  }
  bool IsCaptured() const { return graph_exec_ != nullptr; }

  bool IsCapturing() {
    cudaStreamCaptureStatus status;
    OF_CUDA_CHECK(cudaStreamIsCapturing(stream_, &status));
    return status != cudaStreamCaptureStatusNone;
  };

  void BeginCapture() {
    OF_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  }

  void EndCapture() {
    cudaGraph_t graph;
    OF_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphExecUpdateResult update_result;
    cudaGraphNode_t error_node;
    if (graph_exec_ != nullptr) {
      OF_CUDA_CHECK(cudaGraphExecUpdate(graph_exec_, graph, &error_node, &update_result));
    }
    if (graph_exec_ == nullptr || update_result != cudaGraphExecUpdateSuccess) {
      if (graph_exec_ != nullptr) { OF_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_)); }
      OF_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, NULL, NULL, 0));
    }
    OF_CUDA_CHECK(cudaGraphDestroy(graph));
  }

  void Launch() { OF_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_)); }

 private:
  cudaStream_t stream_;
  cudaGraphExec_t graph_exec_;
};

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11000

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_GRAPH_CONTEXT_H_
