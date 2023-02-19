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

#define OVERLOAD_BINARY_OP(op)                                                         \
  Dim Dim::operator op(const Dim& other) const {                                       \
    if (this->is_known() && other.is_known()) { return this->value_ op other.value_; } \
    return Unknown();                                                                  \
  }                                                                                    \
  Dim Dim::operator op(int other) const {                                              \
    if (this->is_known()) { return this->value_ op other; }                            \
    return Unknown();                                                                  \
  }                                                                                    \
  Dim Dim::operator op(size_t other) const {                                           \
    if (this->is_known()) { return this->value_ op other; }                            \
    return Unknown();                                                                  \
  }                                                                                    \
  Dim Dim::operator op(int64_t other) const {                                          \
    if (this->is_known()) { return this->value_ op other; }                            \
    return Unknown();                                                                  \
  }

OVERLOAD_BINARY_OP(+)
OVERLOAD_BINARY_OP(-)
OVERLOAD_BINARY_OP(*)
OVERLOAD_BINARY_OP(/)
#undef OVERLOAD_BINARY_OP

#define OVERLOAD_ASSIGN_OP(op)                       \
  Dim& Dim::operator op(const Dim& other) {          \
    if (this->is_known() && other.is_known()) {      \
      this->value_ op other.value_;                  \
    } else {                                         \
      this->value_ = Unknown().value_;               \
    }                                                \
    return *this;                                    \
  }                                                  \
  Dim& Dim::operator op(int other) {                 \
    if (this->is_known()) { this->value_ op other; } \
    return *this;                                    \
  }                                                  \
  Dim& Dim::operator op(size_t other) {              \
    if (this->is_known()) { this->value_ op other; } \
    return *this;                                    \
  }                                                  \
  Dim& Dim::operator op(int64_t other) {             \
    if (this->is_known()) { this->value_ op other; } \
    return *this;                                    \
  }

OVERLOAD_ASSIGN_OP(+=)
OVERLOAD_ASSIGN_OP(-=)
OVERLOAD_ASSIGN_OP(*=)
OVERLOAD_ASSIGN_OP(/=)

#define OVERLOAD_COMPARISON_OP(op)                                     \
  bool operator op(const Dim& a, const Dim& b) {                       \
    if (a.is_known() && b.is_known()) { return a.value_ op b.value_; } \
    return false;                                                      \
  }                                                                    \
  bool operator op(const Dim& a, int b) {                              \
    if (a.is_known()) { return a.value_ op b; }                        \
    return false;                                                      \
  }                                                                    \
  bool operator op(const Dim& a, size_t b) {                           \
    if (a.is_known()) { return a.value_ op b; }                        \
    return false;                                                      \
  }                                                                    \
  bool operator op(const Dim& a, int64_t b) {                          \
    if (a.is_known()) { return a.value_ op b; }                        \
    return false;                                                      \
  }                                                                    \
  bool operator op(int a, const Dim& b) {                              \
    if (b.is_known()) { return a op b.value_; }                        \
    return false;                                                      \
  }                                                                    \
  bool operator op(size_t a, const Dim& b) {                           \
    if (b.is_known()) { return a op b.value_; }                        \
    return false;                                                      \
  }                                                                    \
  bool operator op(int64_t a, const Dim& b) {                          \
    if (b.is_known()) { return a op b.value_; }                        \
    return false;                                                      \
  }

OVERLOAD_COMPARISON_OP(==)
OVERLOAD_COMPARISON_OP(!=)
OVERLOAD_COMPARISON_OP(<)
OVERLOAD_COMPARISON_OP(<=)
OVERLOAD_COMPARISON_OP(>)
OVERLOAD_COMPARISON_OP(>=)
#undef OVERLOAD_COMPARISON_OP

}  // namespace oneflow
