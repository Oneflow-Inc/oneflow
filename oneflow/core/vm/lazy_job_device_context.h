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
#ifndef ONEFLOW_CORE_VM_LAZY_JOB_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_VM_LAZY_JOB_DEVICE_CONTEXT_H_

#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

namespace vm {

class LazyJobDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyJobDeviceCtx);
  LazyJobDeviceCtx() = default;
  ~LazyJobDeviceCtx() override = default;

#ifdef WITH_CUDA
  cudaStream_t cuda_stream() const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  cublasHandle_t cublas_handle() const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  cudnnHandle_t cudnn_handle() const override {
    UNIMPLEMENTED();
    return nullptr;
  }
#endif

  vm::Allocator* mut_allocator() override { return (vm::Allocator*)nullptr; }

  DeviceType device_type() const override {
    UNIMPLEMENTED();
    return DeviceType::kInvalidDevice;
  }

  ep::Stream* stream() override {
    UNIMPLEMENTED();
    return nullptr;
  }

  std::queue<std::weak_ptr<NNGraphIf>>* mut_queue() { return &queue_; }
  std::mutex* mut_mutex() { return &mutex_; }
  std::condition_variable* mut_cond() { return &cond_; }

  void WaitUntilQueueEmptyIfFrontNNGraphNotEquals(const std::shared_ptr<NNGraphIf>& nn_graph) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) { return; }
    const auto& last_nn_graph = queue_.front().lock();
    if (!last_nn_graph) { return; }
    if (last_nn_graph == nn_graph) { return; }
    cond_.wait(lock, [this]() { return queue_.empty(); });
  }

  void EnqueueNNGraph(const std::shared_ptr<NNGraphIf>& nn_graph) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.emplace(nn_graph);
  }

  void DequeueNNGraph() {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.pop();
    cond_.notify_all();
  }

 private:
  std::queue<std::weak_ptr<NNGraphIf>> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LAZY_JOB_DEVICE_CONTEXT_H_
