#ifndef ONEFLOW_CORE_EAGER_LAZY_JOB_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_EAGER_LAZY_JOB_DEVICE_CONTEXT_H_

#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

namespace vm {

class LazyJobDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyJobDeviceCtx);
  LazyJobDeviceCtx() = default;
  ~LazyJobDeviceCtx() override = default;

  const cudaStream_t& cuda_stream() const override { 
    UNIMPLEMENTED();
    return *(const cudaStream_t*)nullptr;
  }
  const cublasHandle_t& cublas_pmh_handle() const override {
    UNIMPLEMENTED();
    return *(const cublasHandle_t*)nullptr;
  }
  const cublasHandle_t& cublas_tensor_op_math_handle() const override {
    UNIMPLEMENTED();
    return *(const cublasHandle_t*)nullptr;
  }
  const cublasHandle_t& cublas_pmd_handle() const override {
    UNIMPLEMENTED();
    return *(const cublasHandle_t*)nullptr;
  }
  const cudnnHandle_t& cudnn_handle() const override {
    UNIMPLEMENTED();
    return *(const cudnnHandle_t*)nullptr;
  }

  void SyncDevice() override { UNIMPLEMENTED(); }

  void AddCallBack(std::function<void()> callback) const override { UNIMPLEMENTED(); }

  vm::Allocator* mut_allocator() override {
    UNIMPLEMENTED();
    return (vm::Allocator*)nullptr;
  }

  std::queue<T>* mut_queue() { return &queue_; }
  std::mutex* mut_mutex() { return &mutex_; }
  std::condition_variable* mut_cond() { return &cond_; }

  void WaitUntilQeueEmptyIfFrontNNGraphNotEquals(const std::shared_ptr<NNGraphIf>& nn_graph)  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) { return; }
    const auto& last_nn_graph = queue->front().lock();
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

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_DEVICE_CONTEXT_H_
