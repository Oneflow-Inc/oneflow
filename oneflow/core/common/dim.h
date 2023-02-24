#pragma once

#include <cstdint>
#include <ostream>
#include <variant>
#include "oneflow/core/common/shape.pb.h"

#include <glog/logging.h>

namespace oneflow {

class SymbolicScalar {};

class Dim {
 private:
  int64_t value_;
  const static int64_t kUnknownDimValue = 1LL << 63;

 public:
  Dim() : value_(kUnknownDimValue) {}
  explicit Dim(const DimProto& proto)
      : value_(proto.has_int64_value() ? proto.int64_value() : kUnknownDimValue) {}
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
  int64_t val_or(int64_t default_value) const {
    return is_known() ? value_ : default_value;
  }
  void ToProto(DimProto* proto) const;

#define DECLARE_BINARY_OP(op)                      \
  Dim operator op(const Dim& other) const;         \
  Dim operator op(char other) const;               \
  Dim operator op(unsigned char other) const;      \
  Dim operator op(int other) const;                \
  Dim operator op(unsigned int other) const;       \
  Dim operator op(long other) const;               \
  Dim operator op(unsigned long other) const;      \
  Dim operator op(long long other) const;          \
  Dim operator op(unsigned long long other) const; \
  Dim operator op(float other) const;              \
  Dim operator op(double other) const;

  DECLARE_BINARY_OP(+)
  DECLARE_BINARY_OP(-)
  DECLARE_BINARY_OP(*)
  DECLARE_BINARY_OP(/)
#undef DECLARE_BINARY_OP

#define DECLARE_ASSIGN_OP(op)                 \
  Dim& operator op(const Dim& other);         \
  Dim& operator op(char other);               \
  Dim& operator op(unsigned char other);      \
  Dim& operator op(int other);                \
  Dim& operator op(unsigned int other);       \
  Dim& operator op(long other);               \
  Dim& operator op(unsigned long other);      \
  Dim& operator op(long long other);          \
  Dim& operator op(unsigned long long other); \
  Dim& operator op(float other);              \
  Dim& operator op(double other);

  DECLARE_ASSIGN_OP(+=)
  DECLARE_ASSIGN_OP(-=)
  DECLARE_ASSIGN_OP(*=)
  DECLARE_ASSIGN_OP(/=)

#define DECLARE_COMPARISON_OP_FRIREND(op)                      \
  friend bool operator op(const Dim& a, const Dim& b);         \
  friend bool operator op(const Dim& a, char b);               \
  friend bool operator op(const Dim& a, unsigned char b);      \
  friend bool operator op(const Dim& a, int b);                \
  friend bool operator op(const Dim& a, unsigned int b);       \
  friend bool operator op(const Dim& a, long b);               \
  friend bool operator op(const Dim& a, unsigned long b);      \
  friend bool operator op(const Dim& a, long long b);          \
  friend bool operator op(const Dim& a, unsigned long long b); \
  friend bool operator op(const Dim& a, float b);              \
  friend bool operator op(const Dim& a, double b);             \
  friend bool operator op(char a, const Dim& b);               \
  friend bool operator op(unsigned char a, const Dim& b);      \
  friend bool operator op(int a, const Dim& b);                \
  friend bool operator op(unsigned int a, const Dim& b);       \
  friend bool operator op(long a, const Dim& b);               \
  friend bool operator op(unsigned long a, const Dim& b);      \
  friend bool operator op(long long a, const Dim& b);          \
  friend bool operator op(unsigned long long a, const Dim& b); \
  friend bool operator op(float a, const Dim& b);              \
  friend bool operator op(double a, const Dim& b);

  DECLARE_COMPARISON_OP_FRIREND(==)
  DECLARE_COMPARISON_OP_FRIREND(!=)
  DECLARE_COMPARISON_OP_FRIREND(<)
  DECLARE_COMPARISON_OP_FRIREND(<=)
  DECLARE_COMPARISON_OP_FRIREND(>)
  DECLARE_COMPARISON_OP_FRIREND(>=)
};

static_assert(sizeof(Dim) == sizeof(int64_t), "");

std::ostream& operator<<(std::ostream& os, const Dim& dim);

#define DECLARE_COMPARISON_OP(op)                       \
  bool operator op(const Dim& a, const Dim& b);         \
  bool operator op(const Dim& a, char b);               \
  bool operator op(const Dim& a, unsigned char b);      \
  bool operator op(const Dim& a, int b);                \
  bool operator op(const Dim& a, unsigned int b);       \
  bool operator op(const Dim& a, long b);               \
  bool operator op(const Dim& a, unsigned long b);      \
  bool operator op(const Dim& a, long long b);          \
  bool operator op(const Dim& a, unsigned long long b); \
  bool operator op(const Dim& a, float b);              \
  bool operator op(const Dim& a, double b);             \
  bool operator op(char a, const Dim& b);               \
  bool operator op(unsigned char a, const Dim& b);      \
  bool operator op(int a, const Dim& b);                \
  bool operator op(unsigned int a, const Dim& b);       \
  bool operator op(long a, const Dim& b);               \
  bool operator op(unsigned long a, const Dim& b);      \
  bool operator op(long long a, const Dim& b);          \
  bool operator op(unsigned long long a, const Dim& b); \
  bool operator op(float a, const Dim& b);              \
  bool operator op(double a, const Dim& b);

DECLARE_COMPARISON_OP(==)
DECLARE_COMPARISON_OP(!=)
DECLARE_COMPARISON_OP(<)
DECLARE_COMPARISON_OP(<=)
DECLARE_COMPARISON_OP(>)
DECLARE_COMPARISON_OP(>=)

#undef DECLARE_COMPARISON_OP

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::Dim> {
  size_t operator()(const oneflow::Dim& dim) const {
    return std::hash<int64_t>()(*reinterpret_cast<const int64_t*>(&dim));
  }
};

}  // namespace std
