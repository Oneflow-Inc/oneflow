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
#ifndef ONEFLOW_CORE_EP_STREAM_H_
#define ONEFLOW_CORE_EP_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/event.h"

namespace oneflow {

namespace ep {

class Device;

class Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Stream);
  Stream() = default;
  virtual ~Stream() = default;

  virtual DeviceType device_type() const = 0;
  virtual Device* device() const = 0;
  virtual Maybe<void> Sync() = 0;
  virtual void RecordEvent(Event* event) = 0;
  virtual Maybe<void> GetAsyncError() { return Maybe<void>::Ok(); }

  virtual Maybe<void> AllocAsync(void** ptr, size_t size) { UNIMPLEMENTED_THEN_RETURN(); }
  virtual Maybe<void> FreeAsync(void* ptr) { UNIMPLEMENTED_THEN_RETURN(); }
  template<typename T>
  Maybe<void> AllocAsync(T** ptr, size_t size) {
    return AllocAsync(reinterpret_cast<void**>(ptr), size);
  }

  virtual Maybe<void> OnExecutionContextSetup() { return Maybe<void>::Ok(); }
  virtual Maybe<void> OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

  template<typename T>
  T* As() {
    return static_cast<T*>(this);
  }
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_STREAM_H_
