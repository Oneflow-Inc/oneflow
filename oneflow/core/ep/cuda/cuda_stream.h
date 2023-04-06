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
#include "oneflow/core/ep/cuda/cuda_device.h"

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

struct CudaLaunchConfig {
  dim3 grid_dim;
  dim3 block_dim;
  size_t shared_mem_size;
  CudaLaunchConfig() : grid_dim{}, block_dim{}, shared_mem_size(0) {}

  CudaLaunchConfig(unsigned int grid_size, unsigned int block_size, size_t shared_mem_size)
      : grid_dim(grid_size), block_dim(block_size), shared_mem_size(shared_mem_size) {}
};

class CudaStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStream);
  explicit CudaStream(CudaDevice* device);
  ~CudaStream() override;

  static constexpr uint32_t kDefaultBlockSize = 256;

  DeviceType device_type() const override;
  CudaDevice* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;
  Maybe<void> GetAsyncError() override;

  Maybe<void> AllocAsync(void** ptr, size_t size) override;
  Maybe<void> FreeAsync(void* ptr) override;

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  cudaStream_t cuda_stream() const;
  cublasHandle_t cublas_handle() const;
#if CUDA_VERSION >= 11000
  cusolverDnHandle_t cusolver_dn_handle() const;
#endif

#if CUDA_VERSION >= 10010

  cublasLtHandle_t cublas_lt_handle() const;

#endif

  cudnnHandle_t cudnn_handle() const;
  void* cublas_workspace() const;
  size_t cublas_workspace_size() const;
  const cudaDeviceProp& device_properties() const;
  int cuda_arch() const;

  void InitLaunchConfigWithWaves(CudaLaunchConfig* config, size_t elem_cnt, size_t block_size,
                                 size_t max_waves) const {
    const uint32_t max_grid_size = max_waves * device_properties().multiProcessorCount
                                   * (device_properties().maxThreadsPerMultiProcessor / block_size);
    const uint32_t grid_size =
        std::min<uint32_t>(max_grid_size, (elem_cnt + block_size - 1) / block_size);
    config->grid_dim = dim3(grid_size);
    config->block_dim = dim3(block_size);
    config->shared_mem_size = 0;
  }

#ifdef __CUDACC__
  template<typename... Params, typename... Args>
  void LaunchKernel(void (*kernel)(Params...), const CudaLaunchConfig& launch_config,
                    Args... args) {
    kernel<<<launch_config.grid_dim, launch_config.block_dim, launch_config.shared_mem_size,
             cuda_stream()>>>(args...);
  }

  template<typename... Params, typename... Args>
  void LaunchKernel(void (*kernel)(Params...), size_t elem_cnt, size_t max_waves, Args... args) {
    constexpr uint32_t block_size = kDefaultBlockSize;
    CudaLaunchConfig config{};
    InitLaunchConfigWithWaves(&config, elem_cnt, block_size, max_waves);
    LaunchKernel(kernel, config, args...);
  }

  template<typename... Params, typename... Args>
  void LaunchKernelDefaultWaves(void (*kernel)(Params...), size_t elem_cnt, Args... args) {
    const size_t default_waves = 32;
    LaunchKernel(kernel, elem_cnt, default_waves, args...);
  }
#endif  // __CUDACC__

#ifdef WITH_CUDA_GRAPHS
  void BeginGraphCapture();
  void EndGraphCapture(CudaGraphExecutable* executable);
  bool IsGraphCapturing() const;
  void LaunchGraph(const CudaGraphExecutable* executable);
#endif  // WITH_CUDA_GRAPHS

 private:
  cudaStream_t cuda_stream_{};
  cublasHandle_t cublas_handle_{};
#if CUDA_VERSION >= 11000
  cusolverDnHandle_t cusolver_dn_handle_{};
#endif

#if CUDA_VERSION >= 10010

  cublasLtHandle_t cublas_lt_handle_{};

#endif

  cudnnHandle_t cudnn_handle_{};
  int device_index_;
  void* workspace_{};
  size_t workspace_size_{};
#ifdef WITH_CUDA_GRAPHS
  bool is_graph_capturing_{};
#endif  // WITH_CUDA_GRAPHS
  CudaDevice* device_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_
