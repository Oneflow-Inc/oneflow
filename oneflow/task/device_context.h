#ifndef _DEVICE_DEVICE_CONTEXT_H_
#define _DEVICE_DEVICE_CONTEXT_H_
#include <memory>
#include <unordered_map>

#include "device/device_alternate.h"
#include "layers/base_layer.h"
#include "common/task_type.h"

/*
DeviceContext contains all the computational resources on a particular device, 
such as the streams, cublasHandle, curandHandle, if it is a GPU device.
*/
/*
NOTE(jiyuan): may need more streams to hold small computation tasks, such as
the case with more deep pipelining stages
*/
/*
May need to create DeviceContext according to what types of tasks will execute
on the device. For example, there are two tasks on a particular devices, e.g., 
a task performing convolution type task as part of some data-parallelism job, 
another task performing inner-produced task as part of some model-parallelism.
There is no direct dependency between these two tasks, we have no reason to 
force them use a same stream, which will implicitly add sequential constraints
between the two. Similarly, the forward pass and the backward pass of a particular
actor may use different streams too, in this way, the actor could work on the 
forward pass of a data batch and simultaneously work on the backward pass of 
another data batch.

There are only two DMA engines for each device. Therefore, for a particular device,
we create a single H2D stream and a single D2H stream, which are shared by all 
the actors on this device. 
TODO(jiyuan):
1, Traverse all the actors and count how many streams should be created on a 
particular device;
2, Name each stream and create a dict from actor to stream, each actor use its own
stream;
*/
namespace oneflow {
template <typename Dtype>
class Task;

class Stream;

class CublasHandle;

class CudnnHandle;

template <typename Dtype>
class DeviceContext {
 public:
  explicit DeviceContext(int32_t physical_id);
  ~DeviceContext();

  void RegisterTask(const Task<Dtype>* task);

  cublasHandle_t cublas_handle() const;

  cudnnHandle_t cudnn_handle() const;

  cudaStream_t cuda_stream(int32_t task_id) const;

 private:
  int32_t physical_id_;

  std::shared_ptr<CublasHandle> cublas_handle_;
  std::shared_ptr<CudnnHandle> cudnn_handle_;
  std::shared_ptr<Stream> stream_h2d_;
  std::shared_ptr<Stream> stream_d2h_;
  std::vector<std::shared_ptr<Stream>> stream_computes_;
  // Mapping from task_id to task-specific stream. So far, each task has a
  // single stream.
  std::unordered_map<int32_t, std::shared_ptr<Stream>> stream_dict_;

  DeviceContext(const DeviceContext& other) = delete;
  DeviceContext& operator=(const DeviceContext& other) = delete;
};

}  // namespace oneflow
#endif  // _DEVICE_DEVICE_CONTEXT_H_
