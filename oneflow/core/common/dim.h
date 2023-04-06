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
#pragma once

#include <cstdint>
#include <ostream>
#include <variant>
#include "oneflow/core/common/shape.pb.h"

#include <glog/logging.h>

namespace oneflow {

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
  int64_t val_or(int64_t default_value) const { return is_known() ? value_ : default_value; }
  void ToProto(DimProto* proto) const;

#define DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES(op)    \
  friend Dim operator op(Dim a, Dim b);                \
  friend Dim operator op(Dim a, char b);               \
  friend Dim operator op(Dim a, unsigned char b);      \
  friend Dim operator op(Dim a, int b);                \
  friend Dim operator op(Dim a, unsigned int b);       \
  friend Dim operator op(Dim a, long b);               \
  friend Dim operator op(Dim a, unsigned long b);      \
  friend Dim operator op(Dim a, long long b);          \
  friend Dim operator op(Dim a, unsigned long long b); \
  friend Dim operator op(char a, Dim b);               \
  friend Dim operator op(unsigned char a, Dim b);      \
  friend Dim operator op(int a, Dim b);                \
  friend Dim operator op(unsigned int a, Dim b);       \
  friend Dim operator op(long a, Dim b);               \
  friend Dim operator op(unsigned long a, Dim b);      \
  friend Dim operator op(long long a, Dim b);          \
  friend Dim operator op(unsigned long long a, Dim b); \
  Dim& operator op##=(Dim other);                      \
  Dim& operator op##=(char other);                     \
  Dim& operator op##=(unsigned char other);            \
  Dim& operator op##=(int other);                      \
  Dim& operator op##=(unsigned int other);             \
  Dim& operator op##=(long other);                     \
  Dim& operator op##=(unsigned long other);            \
  Dim& operator op##=(long long other);                \
  Dim& operator op##=(unsigned long long other);

#define DECLARE_BINARY_OP_FLOATING_TYPES(op) \
  friend Dim operator op(Dim a, float b);    \
  friend Dim operator op(Dim a, double b);   \
  friend Dim operator op(float a, Dim b);    \
  friend Dim operator op(double a, Dim b);   \
  Dim& operator op##=(float other);          \
  Dim& operator op##=(double other);

#define DECLARE_BINARY_OP(op)                 \
  DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES(op) \
  DECLARE_BINARY_OP_FLOATING_TYPES(op)

  DECLARE_BINARY_OP(+)
  DECLARE_BINARY_OP(-)
  DECLARE_BINARY_OP(*)
  DECLARE_BINARY_OP(/)
  DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES(%)
  DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES(|)
  DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES(&)

#undef DECLARE_BINARY_OP
#undef DECLARE_BINARY_OP_FLOATING_TYPES
#undef DECLARE_BINARY_OP_EXCEPT_FLOATING_TYPES

#define DECLARE_COMPARISON_OP(op)                       \
  friend bool operator op(Dim a, Dim b);                \
  friend bool operator op(Dim a, char b);               \
  friend bool operator op(Dim a, unsigned char b);      \
  friend bool operator op(Dim a, int b);                \
  friend bool operator op(Dim a, unsigned int b);       \
  friend bool operator op(Dim a, long b);               \
  friend bool operator op(Dim a, unsigned long b);      \
  friend bool operator op(Dim a, long long b);          \
  friend bool operator op(Dim a, unsigned long long b); \
  friend bool operator op(Dim a, float b);              \
  friend bool operator op(Dim a, double b);             \
  friend bool operator op(char a, Dim b);               \
  friend bool operator op(unsigned char a, Dim b);      \
  friend bool operator op(int a, Dim b);                \
  friend bool operator op(unsigned int a, Dim b);       \
  friend bool operator op(long a, Dim b);               \
  friend bool operator op(unsigned long a, Dim b);      \
  friend bool operator op(long long a, Dim b);          \
  friend bool operator op(unsigned long long a, Dim b); \
  friend bool operator op(float a, Dim b);              \
  friend bool operator op(double a, Dim b);

  DECLARE_COMPARISON_OP(==)
  DECLARE_COMPARISON_OP(!=)
  DECLARE_COMPARISON_OP(<)
  DECLARE_COMPARISON_OP(<=)
  DECLARE_COMPARISON_OP(>)
  DECLARE_COMPARISON_OP(>=)
};

static_assert(sizeof(Dim) == sizeof(int64_t), "");

std::ostream& operator<<(std::ostream& os, Dim dim);

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::Dim> {
  size_t operator()(oneflow::Dim dim) const {
    return std::hash<int64_t>()(*reinterpret_cast<const int64_t*>(&dim));
  }
};

}  // namespace std
