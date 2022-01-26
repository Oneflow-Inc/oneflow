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
#ifndef ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
#define ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_

#include <string>
#include <memory>
#include <sstream>
#include <functional>
#include <utility>

#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace hob {

template<typename Context, typename ValueT>
struct BaseExpr {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
  // NOTE: Performance will be degraded if the destructor is virtual.
  //       So please do NOT implement custom destructor in any child classes of BaseExpr,
  //       and every fields of child classes should be of POD type.
  ~BaseExpr() = default;
#pragma GCC diagnostic pop
  ALWAYS_INLINE virtual scalar_or_const_ref_t<ValueT> get(const Context&) const = 0;
  virtual std::string DebugStr(const Context&, bool display_result = true) const = 0;  // NOLINT
  operator bool() = delete;
};

template<typename Context, typename ValueT, typename E>
struct Expr : public BaseExpr<Context, ValueT> {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
  ~Expr() = default;
#pragma GCC diagnostic pop
};

template<typename Context, typename ValueT>
struct Literal final : public Expr<Context, ValueT, Literal<Context, ValueT>> {
  Literal(const ValueT& val) : Literal(ToString(val), val) {}  // NOLINT
  Literal(const std::string& debug_str, const ValueT& val) : val_(val), debug_str_(debug_str) {}
  ALWAYS_INLINE scalar_or_const_ref_t<ValueT> get(const Context&) const override { return val_; }
  std::string DebugStr(const Context&, bool display_result) const override { return debug_str_; }

 private:
  ValueT val_;
  std::string debug_str_;
};

template<typename Context>
using LiteralBool = Literal<Context, bool>;

template<typename Fn,
         typename Context =
             std::decay_t<typename oneflow::function_traits<Fn>::template arg_type<0>>,
         typename ValueT = std::decay_t<typename oneflow::function_traits<Fn>::return_type>>
struct Custom final : public Expr<Context, ValueT, Custom<Fn>> {
  explicit Custom(Fn fn) : Custom("", fn) {}
  Custom(std::string debug_str, Fn fn) : fn_(std::move(fn)), debug_str_(std::move(debug_str)) {}
  ALWAYS_INLINE scalar_or_const_ref_t<ValueT> get(const Context& context) const override {
    return fn_(context);
  }
  std::string DebugStr(const Context&, bool display_result) const override { return debug_str_; }

 private:
  Fn fn_;
  std::string debug_str_;
};

template<typename Fn>
ALWAYS_INLINE inline Custom<Fn> make_custom(Fn fn) {
  return Custom<Fn>(std::forward<Fn>(fn));
}

template<typename Fn>
ALWAYS_INLINE inline Custom<Fn> make_custom(const std::string& debug_str, Fn fn) {
  return Custom<Fn>(debug_str, std::forward<Fn>(fn));
}

template<typename Context, typename E>
using BoolExpr = Expr<Context, bool, E>;

template<typename Context, typename E>
struct NotBoolFunctor final : public BoolExpr<Context, NotBoolFunctor<Context, E>> {
  explicit NotBoolFunctor(const E& expr) : expr_(expr) {}

  ALWAYS_INLINE bool get(const Context& context) const override { return !expr_.get(context); }

  std::string DebugStr(const Context& ctx, bool display_result) const override {
    std::ostringstream string_stream;
    string_stream << "("
                  << "not " << expr_.DebugStr(ctx, display_result) << ")";
    return string_stream.str();
  }

 private:
  const E expr_;
};

template<typename Context, typename E>
NotBoolFunctor<Context, E> operator!(BoolExpr<Context, E> const& lhs) {
  return NotBoolFunctor<Context, E>(*static_cast<const E*>(&lhs));
}

#define DEFINE_BINARY_FUNCTOR(name, op)                                                           \
  template<typename Context, typename E1, typename E2>                                            \
  struct name##BoolFunctor final : public BoolExpr<Context, name##BoolFunctor<Context, E1, E2>> { \
    name##BoolFunctor(const E1& lhs, const E2& rhs) : lhs_(lhs), rhs_(rhs) {}                     \
                                                                                                  \
    ALWAYS_INLINE bool get(const Context& context) const override;                                \
                                                                                                  \
    std::string DebugStr(const Context& ctx, bool display_result) const override;                 \
                                                                                                  \
   private:                                                                                       \
    const E1 lhs_;                                                                                \
    const E2 rhs_;                                                                                \
  };                                                                                              \
                                                                                                  \
  template<typename Context, typename ValueT, typename E1, typename E2>                           \
  name##BoolFunctor<Context, E1, E2> operator op(Expr<Context, ValueT, E1> const& lhs,            \
                                                 Expr<Context, ValueT, E2> const& rhs) {          \
    return name##BoolFunctor<Context, E1, E2>(*static_cast<const E1*>(&lhs),                      \
                                              *static_cast<const E2*>(&rhs));                     \
  }                                                                                               \
                                                                                                  \
  template<typename Context, typename ValueT, typename E1>                                        \
  name##BoolFunctor<Context, E1, Literal<Context, ValueT>> operator op(                           \
      Expr<Context, ValueT, E1> const& lhs, ValueT const& rhs) {                                  \
    return name##BoolFunctor<Context, E1, Literal<Context, ValueT>>(                              \
        *static_cast<const E1*>(&lhs), Literal<Context, ValueT>(rhs));                            \
  }

DEFINE_BINARY_FUNCTOR(Equal, ==)
DEFINE_BINARY_FUNCTOR(And, &&)
DEFINE_BINARY_FUNCTOR(Or, ||)
DEFINE_BINARY_FUNCTOR(Greater, >)
DEFINE_BINARY_FUNCTOR(Less, <)
DEFINE_BINARY_FUNCTOR(EqualOrGreater, >=)
DEFINE_BINARY_FUNCTOR(EqualOrLess, <=)

#undef DEFINE_BINARY_FUNCTOR

#define DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(name, op)                                  \
  template<typename Context, typename E1, typename E2>                                      \
  ALWAYS_INLINE inline bool name##BoolFunctor<Context, E1, E2>::get(const Context& context) \
      const {                                                                               \
    return lhs_.get(context) op rhs_.get(context);                                          \
  }                                                                                         \
  template<typename Context, typename E1, typename E2>                                      \
  std::string name##BoolFunctor<Context, E1, E2>::DebugStr(const Context& ctx,              \
                                                           bool display_result) const {     \
    std::string l_str = lhs_.DebugStr(ctx, display_result);                                 \
    std::string r_str = rhs_.DebugStr(ctx, display_result);                                 \
    std::ostringstream string_stream;                                                       \
    string_stream << "(" << l_str << " " << OF_PP_STRINGIZE(op) << " " << r_str << ")";     \
    return string_stream.str();                                                             \
  }

DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(Equal, ==)
DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(Greater, >)
DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(Less, <)
DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(EqualOrGreater, >=)
DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS(EqualOrLess, <=)

#undef DEFINE_NON_SHORT_CIRCUIT_FUNCTOR_METHODS

template<typename Context, typename E1, typename E2>
ALWAYS_INLINE inline bool AndBoolFunctor<Context, E1, E2>::get(const Context& context) const {
  bool lhs_result = lhs_.get(context);
  if (!lhs_result) { return false; }
  return rhs_.get(context);
}

template<typename Context, typename E1, typename E2>
std::string AndBoolFunctor<Context, E1, E2>::DebugStr(const Context& ctx,
                                                      bool display_result) const {
  std::string l_str = lhs_.DebugStr(ctx, display_result);
  display_result = display_result && lhs_.get(ctx);
  std::string r_str = rhs_.DebugStr(ctx, display_result);
  std::ostringstream string_stream;
  string_stream << "(" << l_str << " and " << r_str << ")";
  return string_stream.str();
}

template<typename Context, typename E1, typename E2>
ALWAYS_INLINE inline bool OrBoolFunctor<Context, E1, E2>::get(const Context& context) const {
  bool lhs_result = lhs_.get(context);
  if (lhs_result) { return true; }
  return rhs_.get(context);
}

template<typename Context, typename E1, typename E2>
std::string OrBoolFunctor<Context, E1, E2>::DebugStr(const Context& ctx,
                                                     bool display_result) const {
  std::string l_str = lhs_.DebugStr(ctx, display_result);
  display_result = display_result && (!lhs_.get(ctx));
  std::string r_str = rhs_.DebugStr(ctx, display_result);
  std::ostringstream string_stream;
  string_stream << "(" << l_str << " or " << r_str << ")";
  return string_stream.str();
}

template<typename Context, typename E1>
EqualBoolFunctor<Context, E1, Literal<Context, std::string>> operator==(
    Expr<Context, std::string, E1> const& lhs, const char* rhs) {
  return EqualBoolFunctor<Context, E1, Literal<Context, std::string>>(
      *static_cast<const E1*>(&lhs), Literal<Context, std::string>(rhs));
}

}  // namespace hob

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
