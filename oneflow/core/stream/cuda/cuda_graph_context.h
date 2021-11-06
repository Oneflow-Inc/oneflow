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
#ifndef ONEFLOW_CORE_STREAM_CUDA_CUDA_GRAPH_CONTEXT_H_
#define ONEFLOW_CORE_STREAM_CUDA_CUDA_GRAPH_CONTEXT_H_

#include "oneflow/core/common/util.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 11000
#define WITH_CUDA_GRAPHS
#endif  // CUDA_VERSION >= 11000

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA_GRAPHS

class CudaGraphExecutable {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaGraphExecutable);
  CudaGraphExecutable();
  ~CudaGraphExecutable();

  void Update(cudaGraph_t graph);
  void Launch(cudaStream_t stream) const;
  bool IsInstantiated() const;

 private:
  void Reset();

  cudaGraphExec_t graph_exec_;
  int dev_;
};

class CudaGraphContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaGraphContext);
  CudaGraphContext() = default;
  virtual ~CudaGraphContext() = default;

  virtual void BeginGraphCapture() = 0;
  virtual void EndGraphCapture(CudaGraphExecutable* executable) = 0;
  virtual bool IsGraphCapturing() const = 0;
  virtual void LaunchGraph(const CudaGraphExecutable* executable) = 0;
};

class GenericCudaGraphContext : public CudaGraphContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenericCudaGraphContext);
  explicit GenericCudaGraphContext(cudaStream_t stream);
  ~GenericCudaGraphContext() override;

  void BeginGraphCapture() override;
  void EndGraphCapture(CudaGraphExecutable* executable) override;
  bool IsGraphCapturing() const override;
  void LaunchGraph(const CudaGraphExecutable* executable) override;

 private:
  cudaStream_t stream_;
  bool is_graph_capturing_;
};

#endif  // WITH_CUDA_GRAPHS

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_STREAM_CUDA_CUDA_GRAPH_CONTEXT_H_
