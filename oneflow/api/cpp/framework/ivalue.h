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
#ifndef ONEFLOW_API_CPP_FRAMEWORK_IVALUE_H_
#define ONEFLOW_API_CPP_FRAMEWORK_IVALUE_H_

#include <cstdint>
#include <memory>
#include <vector>
#include "tensor.h"

namespace oneflow_api {

class IValue {
 public:
  explicit IValue(int value);
  explicit IValue(int64_t value);
  explicit IValue(double value);
  explicit IValue(bool value);
  explicit IValue(const Tensor& value);
  explicit IValue(Tensor&& value);
  explicit IValue(const std::vector<Tensor>& value);
  explicit IValue(std::vector<Tensor>&& value);
  IValue(const IValue& value);
  ~IValue();

  bool IsInt();
  bool IsDouble();
  bool IsBool();
  bool IsTensor();
  bool IsTensorVector();

  const int64_t ToInt() const;
  const double ToDouble() const;
  const bool ToBool() const;
  const Tensor& ToTensor() const;
  const std::vector<Tensor>& ToTensorVector() const;

 private:
  enum class Tag { kInt = 0, kDouble = 1, kBool = 2, kTensor = 3, kTensorVector = 4 };
  friend std::ostream& operator<<(std::ostream&, const Tag&);

  union Payload {  // NOLINT
    union InternalPayload {
      InternalPayload() : v_int(0) {}

      int64_t v_int;
      double v_double;
      bool v_bool;
    } i;

    Tensor v_tensor;
    std::vector<Tensor> v_tensor_vector;

    Payload() : i() {}
    ~Payload() {}
  };

  Payload payload_;
  Tag tag_;
};

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_FRAMEWORK_IVALUE_H_
