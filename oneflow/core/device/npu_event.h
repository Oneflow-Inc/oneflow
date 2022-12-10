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
#ifndef ONEFLOW_CORE_DEVICE_NPU_EVENT_H_
#define ONEFLOW_CORE_DEVICE_NPU_EVENT_H_

#ifdef WITH_NPU

#include "oneflow/core/device/npu_util.h"
#include "oneflow/core/common/single_thread_obj_pool.h"

namespace oneflow {

class NpuEvent final {
 public:
  NpuEvent(const NpuEvent&) = delete;
  NpuEvent(NpuEvent&&) = delete;

  NpuEvent(int device_id, unsigned int flags);
  ~NpuEvent();

  int device_id() const { return device_id_; }
  bool Query() const;

  aclrtEvent* mut_event() { return &event_; }

 private:
  int device_id_;
  aclrtEvent event_;
};

class NpuEventProvider {
 public:
  NpuEventProvider(const NpuEventProvider&) = delete;
  NpuEventProvider(NpuEventProvider&&) = delete;
  virtual ~NpuEventProvider() = default;

  virtual std::shared_ptr<NpuEvent> GetNpuEventWithFlags(unsigned int flags) = 0;

 protected:
  NpuEventProvider() = default;
};

class QueryNpuEventProvider : public NpuEventProvider {
 public:
  QueryNpuEventProvider(const QueryNpuEventProvider&) = delete;
  QueryNpuEventProvider(QueryNpuEventProvider&&) = delete;
  QueryNpuEventProvider() = default;
  virtual ~QueryNpuEventProvider() = default;

  std::shared_ptr<NpuEvent> GetNpuEvent() {
    return GetNpuEventWithFlags(0|1);//cudaEventBlockingSync | cudaEventDisableTiming);//dck_caution_here 
  }
};

class SingleThreadReusedEventPool {
 public:
  SingleThreadReusedEventPool(const SingleThreadReusedEventPool&) = delete;
  SingleThreadReusedEventPool(SingleThreadReusedEventPool&&) = delete;
  explicit SingleThreadReusedEventPool(int device_id)
      : events_(new SingleThreadPoolType()), device_id_(device_id) {}
  ~SingleThreadReusedEventPool() = default;

  std::shared_ptr<NpuEvent> GetReusedNpuEventWithFlags(unsigned int flags) {
    return events_->make_shared(device_id_, flags);
  }

 private:
  using SingleThreadPoolType =
      obj_pool::SingleThreadObjPool<NpuEvent, obj_pool::kDisableReconstruct>;
  std::shared_ptr<SingleThreadPoolType> events_;
  int device_id_;
};

class SingleThreadQueryNpuEventProvider : public QueryNpuEventProvider,
                                           public SingleThreadReusedEventPool {
 public:
  SingleThreadQueryNpuEventProvider(const SingleThreadQueryNpuEventProvider&) = delete;
  SingleThreadQueryNpuEventProvider(SingleThreadQueryNpuEventProvider&&) = delete;
  explicit SingleThreadQueryNpuEventProvider(int device_id)
      : QueryNpuEventProvider(), SingleThreadReusedEventPool(device_id) {}
  ~SingleThreadQueryNpuEventProvider() = default;

  std::shared_ptr<NpuEvent> GetNpuEventWithFlags(unsigned int flags) override {
    return GetReusedNpuEventWithFlags(flags);
  }
};

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_DEVICE_NPU_EVENT_H_
