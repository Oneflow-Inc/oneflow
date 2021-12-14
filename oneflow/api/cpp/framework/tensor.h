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
#ifndef ONEFLOW_API_CPP_FRAMEWORK_TENSOR_H_
#define ONEFLOW_API_CPP_FRAMEWORK_TENSOR_H_

#include <memory>
#include "device.h"
#include "shape.h"
#include "dtype.h"

namespace oneflow {
namespace one {

class Tensor;

}

}  // namespace oneflow

namespace oneflow_api {

class Tensor final {
  friend class Graph;

 public:
  explicit Tensor(const Shape& shape = Shape(), const Device& device = Device("cpu"),
                  const DType& dtype = DType::kFloat);
  explicit Tensor(const std::shared_ptr<oneflow::one::Tensor>& tensor);

  Tensor(const Tensor& tensor);
  Tensor(Tensor&& tensor) noexcept;

  ~Tensor() = default;

  Tensor& operator=(const Tensor& tensor);
  Tensor& operator=(Tensor&& tensor) noexcept;

  [[nodiscard]] Shape shape() const;
  [[nodiscard]] Device device() const;
  [[nodiscard]] DType dtype() const;

  void zeros_();

  // You should never call __internal_tensor() directly.
  [[nodiscard]] const std::shared_ptr<oneflow::one::Tensor>& __internal_tensor() const;

  template<typename T>
  void copy_to(T* buffer) const;

  [[nodiscard]] static Tensor from_buffer(const void* buffer, const Shape& shape,
                                          const Device& device, const DType& dtype);

 private:
  std::shared_ptr<oneflow::one::Tensor> tensor_ = nullptr;
};

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_FRAMEWORK_TENSOR_H_
