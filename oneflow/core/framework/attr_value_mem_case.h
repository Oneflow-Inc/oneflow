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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MEM_CASE_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MEM_CASE_H_

#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

template<typename T>
class TypeAttrVal final : public AttrValue {
 public:
  TypeAttrVal(T v) : val_(v) {}
  ~TypeAttrVal() = default;

  const T& val() const { return val_; }     
  T& val_not_const() {return val_;}

 private:
  T val_;
};

template<typename T>
const T& AttrValueCast(const AttrValue& val);

template<typename T>
T& AttrValueCastNotConst(AttrValue& val);

}

#endif