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

#ifndef ONEFLOW_CORE_FUNCTIONAL_VALUE_TYPES_H_
#define ONEFLOW_CORE_FUNCTIONAL_VALUE_TYPES_H_

#include <memory>

#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace cfg {
class AttrValue;
}  // namespace cfg

namespace one {

class Tensor;

namespace functional {

enum ValueType {
  kINVALID = 0,
  kVOID,
  kTENSOR,
  kTENSOR_REF,
  kTENSOR_MAYBE,
  kATTR,
  kATTR_REF,
};

#define FUNCTOR_VALUE_TYPE_TRAIT(cpp_type, value_type)                                             \
  template<typename T, typename std::enable_if<std::is_same<T, cpp_type>::value>::type* = nullptr> \
  inline ValueType ValueTypeOf() {                                                                 \
    return value_type;                                                                             \
  }

FUNCTOR_VALUE_TYPE_TRAIT(void, kVOID);
FUNCTOR_VALUE_TYPE_TRAIT(one::Tensor, kTENSOR);
FUNCTOR_VALUE_TYPE_TRAIT(std::shared_ptr<one::Tensor>, kTENSOR_REF);
FUNCTOR_VALUE_TYPE_TRAIT(Maybe<one::Tensor>, kTENSOR_MAYBE);
FUNCTOR_VALUE_TYPE_TRAIT(cfg::AttrValue, kATTR);
FUNCTOR_VALUE_TYPE_TRAIT(std::shared_ptr<cfg::AttrValue>, kATTR_REF);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_VALUE_TYPES_H_
