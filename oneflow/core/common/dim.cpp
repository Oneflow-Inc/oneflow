#include "oneflow/core/common/dim.h"

namespace oneflow {

bool Dim::is_known() const { return value_ != kUnknownDimValue; }

std::ostream& operator<<(std::ostream& os, const Dim& dim) {
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

#define OVERLOAD_BINARY_OP_TYPE(op, type)                        \
  Dim Dim::operator op(type other) const {                       \
    if (this->is_known()) { return Dim(this->value_ op other); } \
    return Unknown();                                            \
  }

#define OVERLOAD_BINARY_OP(op)                                                         \
  Dim Dim::operator op(const Dim& other) const {                                       \
    if (this->is_known() && other.is_known()) { return this->value_ op other.value_; } \
    return Unknown();                                                                  \
  }                                                                                    \
  OVERLOAD_BINARY_OP_TYPE(op, float)                                                   \
  OVERLOAD_BINARY_OP_TYPE(op, double)                                                  \
  OVERLOAD_BINARY_OP_TYPE(op, char)                                                    \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned char)                                           \
  OVERLOAD_BINARY_OP_TYPE(op, int)                                                     \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned int)                                            \
  OVERLOAD_BINARY_OP_TYPE(op, long)                                                    \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned long)                                           \
  OVERLOAD_BINARY_OP_TYPE(op, long long)                                               \
  OVERLOAD_BINARY_OP_TYPE(op, unsigned long long)

OVERLOAD_BINARY_OP(+)
OVERLOAD_BINARY_OP(-)
OVERLOAD_BINARY_OP(*)
OVERLOAD_BINARY_OP(/)

#undef OVERLOAD_BINARY_OP_TYPE
#undef OVERLOAD_BINARY_OP

#define OVERLOAD_ASSIGN_OP_TYPE(op, type)            \
  Dim& Dim::operator op(type other) {                \
    if (this->is_known()) { this->value_ op other; } \
    return *this;                                    \
  }

#define OVERLOAD_ASSIGN_OP(op)                  \
  Dim& Dim::operator op(const Dim& other) {     \
    if (this->is_known() && other.is_known()) { \
      this->value_ op other.value_;             \
    } else {                                    \
      this->value_ = Unknown().value_;          \
    }                                           \
    return *this;                               \
  }                                             \
  OVERLOAD_ASSIGN_OP_TYPE(op, float)            \
  OVERLOAD_ASSIGN_OP_TYPE(op, double)           \
  OVERLOAD_ASSIGN_OP_TYPE(op, char)             \
  OVERLOAD_ASSIGN_OP_TYPE(op, unsigned char)    \
  OVERLOAD_ASSIGN_OP_TYPE(op, int)              \
  OVERLOAD_ASSIGN_OP_TYPE(op, unsigned int)     \
  OVERLOAD_ASSIGN_OP_TYPE(op, long)             \
  OVERLOAD_ASSIGN_OP_TYPE(op, unsigned long)    \
  OVERLOAD_ASSIGN_OP_TYPE(op, long long)        \
  OVERLOAD_ASSIGN_OP_TYPE(op, unsigned long long)

OVERLOAD_ASSIGN_OP(+=)
OVERLOAD_ASSIGN_OP(-=)
OVERLOAD_ASSIGN_OP(*=)
OVERLOAD_ASSIGN_OP(/=)

#undef OVERLOAD_ASSIGN_OP_TYPE
#undef OVERLOAD_ASSIGN_OP

#define OVERLOAD_COMPARISON_WITH_SCALAR_TYPE(op, fallback_value, type) \
  bool operator op(const Dim& a, type b) {                    \
    if (a.is_known()) { return a.value_ op b; }               \
    return fallback_value;                                    \
  }                                                           \
  bool operator op(type a, const Dim& b) {                    \
    if (b.is_known()) { return a op b.value_; }               \
    return fallback_value;                                    \
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

bool operator==(const Dim& a, const Dim& b) {
  if (a.is_known() && b.is_known()) { return a.value_ == b.value_; }
  // reflexivity: Dim::Unknown() == Dim::Unknown()
  if (!a.is_known() && !b.is_known()) { return true; }
  return false;
}

bool operator!=(const Dim& a, const Dim& b) {
  return !(a == b);
}

#define OVERLOAD_COMPARISON_BETWEEN_DIMS(op)  \
bool operator op(const Dim& a, const Dim& b) {  \
  if (a.is_known() && b.is_known()) { return a.value_ op b.value_; }  \
  return false; \
}

OVERLOAD_COMPARISON_BETWEEN_DIMS(>);
OVERLOAD_COMPARISON_BETWEEN_DIMS(<);
// Unfortunately we cannot hold reflexivity for >= and <=
OVERLOAD_COMPARISON_BETWEEN_DIMS(>=);
OVERLOAD_COMPARISON_BETWEEN_DIMS(<=);

#undef OVERLOAD_COMPARISON_BETWEEN_DIMS

}  // namespace oneflow
