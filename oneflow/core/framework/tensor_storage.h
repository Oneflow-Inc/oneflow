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

namespace oneflow {

namespace eager {

class TensorBuffer;

}

namespace one {

class TensorStorage final {
 public:
  TensorStorage();
  ~TensorStorage() {
    if (releaser_) { (*releaser_)(buffer_); }
  }

  using ReleaserT = std::function<void(const std::shared_ptr<eager::TensorBuffer>&)>;

  const std::shared_ptr<eager::TensorBuffer> buffer() const { return buffer_; }

  void set_releaser(const ReleaserT& releaser) {
    releaser_ = std::make_shared<ReleaserT>(releaser);
  }

 private:
  std::shared_ptr<eager::TensorBuffer> buffer_;
  std::shared_ptr<ReleaserT> releaser_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_STORAGE_H_
