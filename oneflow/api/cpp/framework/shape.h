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
#ifndef ONEFLOW_API_CPP_FRAMEWORK_SHAPE_H_
#define ONEFLOW_API_CPP_FRAMEWORK_SHAPE_H_

#include <memory>
#include <vector>

namespace oneflow {

class Shape;

}

namespace oneflow_api {

class Shape final {
  friend class Tensor;

 public:
  Shape();
  explicit Shape(const std::vector<int64_t>& dim_vec);
  Shape(const std::initializer_list<int64_t>& dim_vec);
  ~Shape() = default;
  Shape& operator=(const Shape& shape);

  [[nodiscard]] bool operator==(const Shape& rhs) const;
  [[nodiscard]] bool operator!=(const Shape& rhs) const;

  void Set(int64_t index, int64_t val);

  [[nodiscard]] int64_t elem_cnt() const;
  [[nodiscard]] int64_t At(int64_t index) const;
  [[nodiscard]] int64_t NumAxes() const;
  [[nodiscard]] int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  [[nodiscard]] int64_t Count(int64_t begin_axis) const;

 private:
  std::shared_ptr<oneflow::Shape> shape_ = nullptr;

  friend std::ostream& operator<<(std::ostream&, const Shape&);
};
}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_FRAMEWORK_SHAPE_H_
