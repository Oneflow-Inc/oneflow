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

class AttrMap;
namespace cfg {
class AttrValue;
}  // namespace cfg

namespace one {
class Tensor;
class TensorTuple;

namespace functional {
class Scalar;
}  // namespace functional
}  // namespace one

namespace one {
namespace functional {

enum ValueType {
  kINVALID = 0,
  kVOID,
  kINT32,
  kUINT32,
  kINT64,
  kUINT64,
  kFLOAT,
  kDOUBLE,
  kBOOL,
  kINT32_LIST,
  kUINT32_LIST,
  kINT64_LIST,
  kUINT64_LIST,
  kFLOAT_LIST,
  kDOUBLE_LIST,
  kBOOL_LIST,
  kSCALAR,
  kTENSOR,
  kTENSOR_REF,
  kTENSOR_MAYBE,
  kTENSOR_TUPLE,
  kTENSOR_TUPLE_REF,
  kTENSOR_TUPLE_MAYBE,
  kATTR,
  kATTR_REF,
  kATTR_MAP,
};

#define FUNCTOR_VALUE_TYPE_TRAIT(cpp_type, value_type)                                           \
  template<typename T, typename std::enable_if<std::is_same<T, cpp_type>::value, int>::type = 0> \
  inline ValueType ValueTypeOf() {                                                               \
    return value_type;                                                                           \
  }

FUNCTOR_VALUE_TYPE_TRAIT(void, kVOID);
FUNCTOR_VALUE_TYPE_TRAIT(int32_t, kINT32);
FUNCTOR_VALUE_TYPE_TRAIT(uint32_t, kUINT32);
FUNCTOR_VALUE_TYPE_TRAIT(int64_t, kINT64);
FUNCTOR_VALUE_TYPE_TRAIT(uint64_t, kUINT64);
FUNCTOR_VALUE_TYPE_TRAIT(float, kFLOAT);
FUNCTOR_VALUE_TYPE_TRAIT(double, kDOUBLE);
FUNCTOR_VALUE_TYPE_TRAIT(bool, kBOOL);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<int32_t>, kINT32_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<uint32_t>, kUINT32_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<int64_t>, kINT64_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<uint64_t>, kUINT64_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<float>, kFLOAT_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<double>, kDOUBLE_LIST);
FUNCTOR_VALUE_TYPE_TRAIT(std::vector<bool>, kBOOL_LIST);

FUNCTOR_VALUE_TYPE_TRAIT(Scalar, kSCALAR);
FUNCTOR_VALUE_TYPE_TRAIT(one::Tensor, kTENSOR);
FUNCTOR_VALUE_TYPE_TRAIT(std::shared_ptr<one::Tensor>, kTENSOR_REF);
FUNCTOR_VALUE_TYPE_TRAIT(Maybe<one::Tensor>, kTENSOR_MAYBE);
FUNCTOR_VALUE_TYPE_TRAIT(one::TensorTuple, kTENSOR_TUPLE);
FUNCTOR_VALUE_TYPE_TRAIT(std::shared_ptr<one::TensorTuple>, kTENSOR_TUPLE_REF);
FUNCTOR_VALUE_TYPE_TRAIT(Maybe<one::TensorTuple>, kTENSOR_TUPLE_MAYBE);
FUNCTOR_VALUE_TYPE_TRAIT(cfg::AttrValue, kATTR);
FUNCTOR_VALUE_TYPE_TRAIT(std::shared_ptr<cfg::AttrValue>, kATTR_REF);
FUNCTOR_VALUE_TYPE_TRAIT(AttrMap, kATTR_MAP);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_VALUE_TYPES_H_
