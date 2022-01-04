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
#include "oneflow/api/cpp/framework/ivalue.h"
#include <glog/logging.h>

namespace oneflow_api {

namespace of = oneflow;

std::ostream& operator<<(std::ostream& os, const IValue::Tag& tag) {
  os << static_cast<int>(tag);
  return os;
}

int64_t IValue::ToInt() const {
  CHECK_EQ(tag_, Tag::kInt) << "Current value is not an int.";
  return payload_.i.v_int;
}

double IValue::ToDouble() const {
  CHECK_EQ(tag_, Tag::kDouble) << "Current value is not a double.";
  return payload_.i.v_double;
}

bool IValue::ToBool() const {
  CHECK_EQ(tag_, Tag::kBool) << "Current value is not a bool.";
  return payload_.i.v_bool;
}

const Tensor& IValue::ToTensor() const {
  CHECK_EQ(tag_, Tag::kTensor) << "Current value is not a tensor.";
  return payload_.v_tensor;
}

const std::vector<Tensor>& IValue::ToTensorVector() const {
  CHECK_EQ(tag_, Tag::kTensorVector) << "Current value is not a vector of tensor.";
  return payload_.v_tensor_vector;
}

}  // namespace oneflow_api
