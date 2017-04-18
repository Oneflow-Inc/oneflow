#include "task/device_context.h"
#include <memory>
#include "common/common.h"
#include "context/one.h"
#include "context/id_map.h"
#include "task/task.h"
#include "device/device_descriptor.h"
#include "device/device_resource.h"

namespace oneflow {
template <typename Dtype>
DeviceContext<Dtype>::DeviceContext(int32_t physical_id)
  : physical_id_(physical_id),
  cudnn_handle_(NULL),
  cublas_handle_(NULL) {
}

template <typename Dtype>
DeviceContext<Dtype>::~DeviceContext() {
  // Set device context before releasing cuda-related resources
  SetCurrentDevice(physical_id_);
}

template <typename Dtype>
void DeviceContext<Dtype>::RegisterTask(const Task<Dtype>* task) {
  //FixMe xiaoshu
}

template <typename Dtype>
cublasHandle_t DeviceContext<Dtype>::cublas_handle() const {
  return cublas_handle_->get_cublas_handle();
}

template <typename Dtype>
cudnnHandle_t DeviceContext<Dtype>::cudnn_handle() const {
  return cudnn_handle_->get_cudnn_handle();
}

template <typename Dtype>
cudaStream_t DeviceContext<Dtype>::cuda_stream(int32_t task_id) const {
  auto stream_it = stream_dict_.find(task_id);
  CHECK(stream_it != stream_dict_.end());
  return stream_it->second->get_cuda_stream();
}

INSTANTIATE_CLASS(DeviceContext);
}  // namespace oneflow
