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
#ifndef ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_
#define ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_

#include "oneflow/core/ep/include/stream.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 11000
#define WITH_CUDA_GRAPHS
#endif  // CUDA_VERSION >= 11000

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace ep {

class CudaDevice;

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

#endif  // WITH_CUDA_GRAPHS

class CudaStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStream);
  explicit CudaStream(CudaDevice* device);
  ~CudaStream() override;

  DeviceType device_type() const override;
  Device* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  cudaStream_t cuda_stream() const;
  cublasHandle_t cublas_handle() const;
  cudnnHandle_t cudnn_handle() const;
  const cudaDeviceProp& device_properties() const;

#ifdef WITH_CUDA_GRAPHS
  void BeginGraphCapture();
  void EndGraphCapture(CudaGraphExecutable* executable);
  bool IsGraphCapturing() const;
  void LaunchGraph(const CudaGraphExecutable* executable);
#endif  // WITH_CUDA_GRAPHS

 private:
  cudaStream_t cuda_stream_{};
  cublasHandle_t cublas_handle_{};
  cudnnHandle_t cudnn_handle_{};
  int device_index_;
#if CUBLAS_VERSION >= 11200
  void* workspace_{};
  size_t workspace_size_{};
#endif  // CUBLAS_VERSION >= 11200
#ifdef WITH_CUDA_GRAPHS
  bool is_graph_capturing_{};
#endif  // WITH_CUDA_GRAPHS
  CudaDevice* device_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_
