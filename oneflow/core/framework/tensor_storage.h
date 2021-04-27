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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_STORAGE_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_STORAGE_H_

#include <memory>
#include <functional>

namespace oneflow {

class ParallelDesc;

namespace vm {

class TensorBuffer;

}  // namespace vm

namespace one {

class TensorStorage final {
 public:
  explicit TensorStorage(const std::shared_ptr<const ParallelDesc>& parallel_desc);
  explicit TensorStorage(const std::shared_ptr<vm::TensorBuffer>& tensor_buffer);
  ~TensorStorage();

  using ReleaserHookT = std::function<void(const std::shared_ptr<vm::TensorBuffer>&)>;

  const std::shared_ptr<vm::TensorBuffer> buffer() const { return buffer_; }

  void set_releaser_hook(const ReleaserHookT& releaser_hook) {
    releaser_hook_ = std::make_shared<ReleaserHookT>(releaser_hook);
  }

 private:
  std::shared_ptr<vm::TensorBuffer> buffer_;
  std::shared_ptr<ReleaserHookT> releaser_hook_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_STORAGE_H_
