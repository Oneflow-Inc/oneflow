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
#include "oneflow/core/common/dim.h"

namespace oneflow {

bool Dim::is_known() const { return value_ != kUnknownDimValue; }

std::ostream& operator<<(std::ostream& os, Dim dim) {
  if (dim.is_known()) {
    os << static_cast<int64_t>(dim);
  } else {
    os << "Unknown";
  }
  return os;
}

void Dim::ToProto(DimProto* proto) const {
  if (is_known()) {
    proto->set_int64_value(value_);
  } else {
    proto->mutable_unknown();
  }
}

#define OVERLOAD_BINARY_OP_TYPE(op, type)                \
  Dim operator op(Dim a, type b) {                       \
    if (a.is_known()) { return a.value_ op b; }          \
    return Dim::Unknown();                               \
  }                                                      \
  Dim operator op(type a, Dim b) {                       \
    if (b.is_known()) { return a op b.value_; }          \
    return Dim::Unknown();                               \
  }                                                      \
  Dim& Dim::operator op##=(type other) {                 \
    if (this->is_known()) { this->value_ op## = other; } \
    return *this;                                        \
  }

#define OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES(op)            \
  Dim operator op(Dim a, Dim b) {                               \
    if (a.is_known() && b.is_known()) { return a.value_ op b; } \
    return Dim::Unknown();                                      \
  }                                                             \
  Dim& Dim::operator op##=(Dim other) {                         \
    if (this->is_known() && other.is_known()) {                 \
      this->value_ op## = other.value_;                         \
    } else {                                                    \
      this->value_ = Unknown().value_;                          \
    }                                                           \
    return *this;                                               \
  }                                                             \
  OVERLOAD_BINARY_OP_TYPE(op, char)                             \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned char)                    \
  OVERLOAD_BINARY_OP_TYPE(op, int)                              \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned int)                     \
  OVERLOAD_BINARY_OP_TYPE(op, long)                             \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned long)                    \
  OVERLOAD_BINARY_OP_TYPE(op, long long)                        \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned long long)

#define OVERLOAD_BINARY_OP_FLOATING_TYPES(op) \
  OVERLOAD_BINARY_OP_TYPE(op, float)          \
  OVERLOAD_BINARY_OP_TYPE(op, double)

#define OVERLOAD_BINARY_OP(op)                 \
  OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES(op) \
  OVERLOAD_BINARY_OP_FLOATING_TYPES(op)

OVERLOAD_BINARY_OP(+)
OVERLOAD_BINARY_OP(-)
OVERLOAD_BINARY_OP(*)
OVERLOAD_BINARY_OP(/)
OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES(%)
OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES(|)
OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES(&)

#undef OVERLOAD_BINARY_OP_EXCEPT_FLOATING_TYPES
#undef OVERLOAD_BINARY_OP_TYPE
#undef OVERLOAD_BINARY_OP

#define OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, type) \
  bool operator op(Dim a, type b) {                                    \
    if (a.is_known()) { return a.value_ op b; }                        \
    return fallback_value;                                             \
  }                                                                    \
  bool operator op(type a, Dim b) {                                    \
    if (b.is_known()) { return a op b.value_; }                        \
    return fallback_value;                                             \
  }

#define OVERLOAD_COMPARISON_WITH_SCALAR(op, fallback_value)               \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, float)         \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, double)        \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, char)          \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, unsigned char) \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, int)           \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, unsigned int)  \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, long)          \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, unsigned long) \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, long long)     \
  OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, unsigned long long)

OVERLOAD_COMPARISON_WITH_SCALAR(==, false)
OVERLOAD_COMPARISON_WITH_SCALAR(!=, true)
OVERLOAD_COMPARISON_WITH_SCALAR(<, false)
OVERLOAD_COMPARISON_WITH_SCALAR(<=, false)
OVERLOAD_COMPARISON_WITH_SCALAR(>, false)
OVERLOAD_COMPARISON_WITH_SCALAR(>=, false)
#undef OVERLOAD_COMPARISON_WITH_SCALAR

bool operator==(Dim a, Dim b) {
  if (a.is_known() && b.is_known()) { return a.value_ == b.value_; }
  // reflexivity: Dim::Unknown() == Dim::Unknown()
  // TODO(daquexian): identify different unknown Dims in the future
  // Example:
  // (1, 3, N, N) --- Conv(h=3, w=2) ---> (1, 3, M, P)
  // (1, 3, N, N) --- Conv(h=3, w=3) ---> (1, 3, M, M)
  // (1, 3, N, N) --- Conv(h=3, w=3, padding=1) ---> (1, 3, N, N)
  // (1, 3, N, N) --- Pooling ---> (1, 3, M, M)
  // (1, 3, N, N) --- ReLU ---> (1, 3, N, N)
  if (!a.is_known() && !b.is_known()) { return true; }
  return false;
}

bool operator!=(Dim a, Dim b) { return !(a == b); }

#define OVERLOAD_COMPARISON_BETWEEN_DIMS(op)                           \
  bool operator op(Dim a, Dim b) {                                     \
    if (a.is_known() && b.is_known()) { return a.value_ op b.value_; } \
    return false;                                                      \
  }

OVERLOAD_COMPARISON_BETWEEN_DIMS(>);
OVERLOAD_COMPARISON_BETWEEN_DIMS(<);
// Unfortunately we cannot hold reflexivity for >= and <=
OVERLOAD_COMPARISON_BETWEEN_DIMS(>=);
OVERLOAD_COMPARISON_BETWEEN_DIMS(<=);

#undef OVERLOAD_COMPARISON_BETWEEN_DIMS

}  // namespace oneflow
