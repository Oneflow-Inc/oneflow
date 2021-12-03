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
  explicit IValue(int value) : tag_(IValue::Tag::kInt) { payload_.i.v_int = value; }

  explicit IValue(int64_t value) : tag_(IValue::Tag::kInt) { payload_.i.v_int = value; }

  explicit IValue(double value) : tag_(IValue::Tag::kDouble) { payload_.i.v_double = value; }

  explicit IValue(bool value) : tag_(IValue::Tag::kBool) { payload_.i.v_bool = value; }

  explicit IValue(const Tensor& value) : tag_(IValue::Tag::kTensor) {
    new (&payload_.v_tensor) Tensor(value);
  }

  explicit IValue(Tensor&& value) : tag_(IValue::Tag::kTensor) {
    new (&payload_.v_tensor) Tensor(std::move(value));
  }

  explicit IValue(const std::vector<Tensor>& value) : tag_(IValue::Tag::kTensorVector) {
    new (&payload_.v_tensor_vector) std::vector<Tensor>(value);
  }

  explicit IValue(std::vector<Tensor>&& value) : tag_(IValue::Tag::kTensorVector) {
    new (&payload_.v_tensor_vector) std::vector<Tensor>(std::move(value));
  }

  IValue(const IValue& value) : tag_(value.tag_) {
    if (IsTensor()) {
      new (&payload_.v_tensor) Tensor(value.ToTensor());
    } else if (IsTensorVector()) {
      new (&payload_.v_tensor_vector) std::vector<Tensor>(value.ToTensorVector());
    } else {
      payload_.i = value.payload_.i;
    }
  }

  ~IValue() {
    if (IsTensor()) { payload_.v_tensor.~Tensor(); }
    if (IsTensorVector()) { payload_.v_tensor_vector.~vector(); }
  }

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
