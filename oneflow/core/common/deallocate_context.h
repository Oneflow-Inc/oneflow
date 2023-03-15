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
#ifndef ONEFLOW_CORE_COMMON_DEALLOCATE_CONTEXT_H_
#define ONEFLOW_CORE_COMMON_DEALLOCATE_CONTEXT_H_

#include <functional>
#include <memory>

namespace oneflow {

class DeallocateContext {
 public:
  DeallocateContext() = default;
  virtual ~DeallocateContext() = default;

  // Support customizing the Dispatch method for releasing a pointer, such as releasing it in a
  // separate thread. Note that T also needs to be a smart pointer.
  template<typename T>
  void Deallocate(std::shared_ptr<T>&& ptr) {
    std::shared_ptr<T> data = ptr;
    ptr.reset();
    // reset shared_ptr inside data by customized Dispatch
    Dispatch([data] { const_cast<std::shared_ptr<T>*>(&data)->reset(); });
    // reset data
    data.reset();
  }

  virtual void Dispatch(std::function<void()> Handle) = 0;
};

class NaiveDeallocateContext final : public DeallocateContext {
 public:
  NaiveDeallocateContext() = default;
  ~NaiveDeallocateContext() = default;

  void Dispatch(std::function<void()> Handle) { Handle(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DEALLOCATE_CONTEXT_H_
