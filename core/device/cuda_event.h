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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_EVENT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_EVENT_H_

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/single_thread_obj_pool.h"

namespace oneflow {

class CudaEvent final {
 public:
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) = delete;

  CudaEvent(int device_id, unsigned int flags);
  ~CudaEvent();

  int device_id() const { return device_id_; }
  bool Query() const;

  cudaEvent_t* mut_event() { return &event_; }

 private:
  int device_id_;
  cudaEvent_t event_;
};

class CudaEventProvider {
 public:
  CudaEventProvider(const CudaEventProvider&) = delete;
  CudaEventProvider(CudaEventProvider&&) = delete;
  virtual ~CudaEventProvider() = default;

  virtual std::shared_ptr<CudaEvent> GetCudaEventWithFlags(unsigned int flags) = 0;

 protected:
  CudaEventProvider() = default;
};

class QueryCudaEventProvider : public CudaEventProvider {
 public:
  QueryCudaEventProvider(const QueryCudaEventProvider&) = delete;
  QueryCudaEventProvider(QueryCudaEventProvider&&) = delete;
  QueryCudaEventProvider() = default;
  virtual ~QueryCudaEventProvider() = default;

  std::shared_ptr<CudaEvent> GetCudaEvent() {
    return GetCudaEventWithFlags(cudaEventBlockingSync | cudaEventDisableTiming);
  }
};

class SingleThreadReusedEventPool {
 public:
  SingleThreadReusedEventPool(const SingleThreadReusedEventPool&) = delete;
  SingleThreadReusedEventPool(SingleThreadReusedEventPool&&) = delete;
  explicit SingleThreadReusedEventPool(int device_id)
      : events_(new SingleThreadPoolType()), device_id_(device_id) {}
  ~SingleThreadReusedEventPool() = default;

  std::shared_ptr<CudaEvent> GetReusedCudaEventWithFlags(unsigned int flags) {
    return events_->make_shared(device_id_, flags);
  }

 private:
  using SingleThreadPoolType =
      obj_pool::SingleThreadObjPool<CudaEvent, obj_pool::kDisableReconstruct>;
  std::shared_ptr<SingleThreadPoolType> events_;
  int device_id_;
};

class SingleThreadQueryCudaEventProvider : public QueryCudaEventProvider,
                                           public SingleThreadReusedEventPool {
 public:
  SingleThreadQueryCudaEventProvider(const SingleThreadQueryCudaEventProvider&) = delete;
  SingleThreadQueryCudaEventProvider(SingleThreadQueryCudaEventProvider&&) = delete;
  explicit SingleThreadQueryCudaEventProvider(int device_id)
      : QueryCudaEventProvider(), SingleThreadReusedEventPool(device_id) {}
  ~SingleThreadQueryCudaEventProvider() = default;

  std::shared_ptr<CudaEvent> GetCudaEventWithFlags(unsigned int flags) override {
    return GetReusedCudaEventWithFlags(flags);
  }
};

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_DEVICE_CUDA_EVENT_H_
