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

  IValue(int32_t value) : tag_(IValue::Tag::kInt32) {  // NOLINT
    payload_.i.v_int = static_cast<int64_t>(value);
  }

  IValue(int64_t value) : tag_(IValue::Tag::kInt64) { payload_.i.v_int = value; }  // NOLINT

  IValue(float value) : tag_(IValue::Tag::kFloat) {  // NOLINT
    payload_.i.v_double = static_cast<double>(value);
  }

  IValue(double value) : tag_(IValue::Tag::kDouble) { payload_.i.v_double = value; }  // NOLINT

  IValue(bool value) : tag_(IValue::Tag::kBool) { payload_.i.v_bool = value; }  // NOLINT

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

  IValue(const std::vector<IValue>& values) : tag_(Tag::kTensorVector) {  // NOLINT
    new (&payload_.v_tensor_vector) std::vector<Tensor>(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      // TODO check type of values[i]
      payload_.v_tensor_vector.at(i) = values.at(i).ToTensor();
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

  explicit operator bool() const { return !IsNone(); }

  ~IValue() { Destory(); }

  bool IsNone() const { return tag_ == Tag::kNone; }

  bool IsInt32() const { return tag_ == Tag::kInt32; }

  bool IsInt64() const { return tag_ == Tag::kInt64; }

  bool IsFloat() const { return tag_ == Tag::kFloat; }

  bool IsDouble() const { return tag_ == Tag::kDouble; }

  bool IsBool() const { return tag_ == Tag::kBool; }

  bool IsTensor() const { return tag_ == Tag::kTensor; }

  bool IsTensorVector() const { return tag_ == Tag::kTensorVector; }

  int32_t ToInt32() const;
  int64_t ToInt64() const;
  float ToFloat() const;
  double ToDouble() const;
  bool ToBool() const;
  const Tensor& ToTensor() const;
  const std::vector<Tensor>& ToTensorVector() const;

 private:
  enum class Tag {
    kNone = 0,
    kInt32 = 1,
    kInt64 = 2,
    kFloat = 3,
    kDouble = 4,
    kBool = 5,
    kTensor = 6,
    kTensorVector = 7
  };
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
