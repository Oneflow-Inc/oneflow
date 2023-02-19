#pragma once

#include <cstdint>
#include <ostream>
#include <variant>

#include <glog/logging.h>

namespace oneflow {

class SymbolicScalar {};

class Dim {
 private:
  int64_t value_;
  const static int64_t kUnknownDimValue = 1LL << 63;

 public:
  Dim() : value_(kUnknownDimValue) {}
  Dim(int64_t value) : value_(value) {}  // NOLINT(google-explicit-constructor)
  static Dim Unknown() {
    static Dim unknown(kUnknownDimValue);
    return unknown;
  }
  bool is_known() const;
  operator int64_t() const {  // NOLINT(google-explicit-constructor)
    CHECK(is_known());
    return value_;
  }
  int64_t* int64_ptr() {
    CHECK(is_known());
    return &value_;
  }
  const int64_t* int64_ptr() const {
    CHECK(is_known());
    return &value_;
  }
  int64_t val() const {
    CHECK(is_known());
    return value_;
  }

#define DECLARE_BINARY_OP(op)              \
  Dim operator op(const Dim& other) const; \
  Dim operator op(int other) const;        \
  Dim operator op(size_t other) const;     \
  Dim operator op(int64_t other) const;

  DECLARE_BINARY_OP(+)
  DECLARE_BINARY_OP(-)
  DECLARE_BINARY_OP(*)
  DECLARE_BINARY_OP(/)
#undef DECLARE_BINARY_OP

#define DECLARE_ASSIGN_OP(op)         \
  Dim& operator op(const Dim& other); \
  Dim& operator op(int other);        \
  Dim& operator op(size_t other);     \
  Dim& operator op(int64_t other);

  DECLARE_ASSIGN_OP(+=)
  DECLARE_ASSIGN_OP(-=)
  DECLARE_ASSIGN_OP(*=)
  DECLARE_ASSIGN_OP(/=)

#define DECLARE_COMPARISON_OP_FRIREND(op)               \
  friend bool operator op(const Dim& a, const Dim& b); \
  friend bool operator op(const Dim& a, int b);        \
  friend bool operator op(const Dim& a, size_t b);     \
  friend bool operator op(const Dim& a, int64_t b);    \
  friend bool operator op(int a, const Dim& b);        \
  friend bool operator op(size_t a, const Dim& b);     \
  friend bool operator op(int64_t a, const Dim& b);

  DECLARE_COMPARISON_OP_FRIREND(==)
  DECLARE_COMPARISON_OP_FRIREND(!=)
  DECLARE_COMPARISON_OP_FRIREND(<)
  DECLARE_COMPARISON_OP_FRIREND(<=)
  DECLARE_COMPARISON_OP_FRIREND(>)
  DECLARE_COMPARISON_OP_FRIREND(>=)
};

static_assert(sizeof(Dim) == sizeof(int64_t), "");

std::ostream& operator<<(std::ostream& os, const Dim& dim);

#define DECLARE_COMPARISON_OP(op)               \
  bool operator op(const Dim& a, const Dim& b); \
  bool operator op(const Dim& a, int b);        \
  bool operator op(const Dim& a, size_t b);     \
  bool operator op(const Dim& a, int64_t b);    \
  bool operator op(int a, const Dim& b);        \
  bool operator op(size_t a, const Dim& b);     \
  bool operator op(int64_t a, const Dim& b);

DECLARE_COMPARISON_OP(==)
DECLARE_COMPARISON_OP(!=)
DECLARE_COMPARISON_OP(<)
DECLARE_COMPARISON_OP(<=)
DECLARE_COMPARISON_OP(>)
DECLARE_COMPARISON_OP(>=)

#undef DECLARE_COMPARISON_OP

}  // namespace oneflow
