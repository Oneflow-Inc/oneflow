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
  IValue() : tag_(IValue::Tag::kNone) {}
  explicit IValue(int value) : tag_(IValue::Tag::kInt) { payload_.i.v_int = value; }

  explicit IValue(int64_t value) : tag_(IValue::Tag::kInt) { payload_.i.v_int = value; }

  explicit IValue(double value) : tag_(IValue::Tag::kDouble) { payload_.i.v_double = value; }

  explicit IValue(bool value) : tag_(IValue::Tag::kBool) { payload_.i.v_bool = value; }

  IValue(const Tensor& value) : tag_(IValue::Tag::kTensor) {  // NOLINT
    new (&payload_.v_tensor) Tensor(value);
  }

  IValue(Tensor&& value) : tag_(IValue::Tag::kTensor) {  // NOLINT
    new (&payload_.v_tensor) Tensor(std::move(value));
  }

  IValue(const std::vector<Tensor>& value) : tag_(IValue::Tag::kTensorVector) {  // NOLINT
    new (&payload_.v_tensor_vector) std::vector<Tensor>(value);
  }

  IValue(std::vector<Tensor>&& value) : tag_(IValue::Tag::kTensorVector) {  // NOLINT
    new (&payload_.v_tensor_vector) std::vector<Tensor>(std::move(value));
  }

  IValue(const IValue& value) : tag_(value.tag_) {
    if (IsTensor()) {
      new (&payload_.v_tensor) Tensor(value.payload_.v_tensor);
    } else if (IsTensorVector()) {
      new (&payload_.v_tensor_vector) std::vector<Tensor>(value.payload_.v_tensor_vector);
    } else {
      payload_.i = value.payload_.i;
    }
  }

  IValue(IValue&& value) noexcept : tag_(value.tag_) { MoveFrom(std::move(value)); }

  IValue& operator=(const IValue& value) {
    if (&value == this) { return *this; }
    this->tag_ = value.tag_;
    *this = IValue(value);
    return *this;
  }

  IValue& operator=(IValue&& value) noexcept {
    if (&value == this) { return *this; }
    Destory();
    this->tag_ = value.tag_;
    MoveFrom(std::move(value));
    return *this;
  }

  ~IValue() { Destory(); }

  bool IsNone() const { return tag_ == Tag::kNone; }

  bool IsInt() const { return tag_ == Tag::kInt; }

  bool IsDouble() const { return tag_ == Tag::kDouble; }

  bool IsBool() const { return tag_ == Tag::kBool; }

  bool IsTensor() const { return tag_ == Tag::kTensor; }

  bool IsTensorVector() const { return tag_ == Tag::kTensorVector; }

  int64_t ToInt() const;
  double ToDouble() const;
  bool ToBool() const;
  const Tensor& ToTensor() const;
  const std::vector<Tensor>& ToTensorVector() const;

 private:
  enum class Tag { kNone = 0, kInt = 1, kDouble = 2, kBool = 3, kTensor = 4, kTensorVector = 5 };
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

  inline void Destory() {
    if (IsTensor()) { payload_.v_tensor.~Tensor(); }
    if (IsTensorVector()) { payload_.v_tensor_vector.~vector(); }
  }

  inline void MoveFrom(IValue&& value) {
    if (IsTensor()) {
      new (&payload_.v_tensor) Tensor(std::move(value.payload_.v_tensor));
    } else if (IsTensorVector()) {
      new (&payload_.v_tensor_vector)
          std::vector<Tensor>(std::move(value.payload_.v_tensor_vector));
    } else {
      payload_.i = value.payload_.i;
    }
    value.ClearToNone();
  }

  inline void ClearToNone() {
    Destory();
    payload_.i.v_int = 0;
    tag_ = Tag::kNone;
  }
};

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_FRAMEWORK_IVALUE_H_
