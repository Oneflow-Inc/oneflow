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
#ifndef ONEFLOW_CORE_VM_EP_EVENT_H_
#define ONEFLOW_CORE_VM_EP_EVENT_H_

#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/include/event.h"
#include "oneflow/core/common/single_thread_obj_pool.h"

namespace oneflow {

class EpEvent final {
 public:
  EpEvent(const EpEvent&) = delete;
  EpEvent(EpEvent&&) = delete;

  EpEvent(ep::Device* device);
  ~EpEvent();

  bool Query() const;

  ep::Device* mut_device() { return device_; }

  ep::Event* mut_event() { return event_; }

 private:
  ep::Device* device_;
  ep::Event* event_;
};

class EpEventProvider {
 public:
  EpEventProvider(const EpEventProvider&) = delete;
  EpEventProvider(EpEventProvider&&) = delete;
  virtual ~EpEventProvider() = default;

  virtual std::shared_ptr<EpEvent> GetReusedEpEvent() = 0;

 protected:
  EpEventProvider() = default;
};

class SingleThreadEpEventProvider final : public EpEventProvider {
 public:
  SingleThreadEpEventProvider(const SingleThreadEpEventProvider&) = delete;
  SingleThreadEpEventProvider(SingleThreadEpEventProvider&&) = delete;
  explicit SingleThreadEpEventProvider(ep::Device* device)
      : EpEventProvider(), events_(new SingleThreadPoolType()), device_(device) {}
  ~SingleThreadEpEventProvider() = default;

  std::shared_ptr<EpEvent> GetReusedEpEvent() override { return events_->make_shared(device_); }

 private:
  using SingleThreadPoolType =
      obj_pool::SingleThreadObjPool<EpEvent, obj_pool::kDisableReconstruct>;
  std::shared_ptr<SingleThreadPoolType> events_;
  ep::Device* device_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_EP_EVENT_H_
