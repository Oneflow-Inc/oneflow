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
#include "oneflow/core/common/to_string.h"

namespace oneflow {

namespace hob {

template<typename Context, typename ValueT>
struct BaseBaseExpr {
  virtual ValueT get(const Context&) const = 0;
  virtual std::string DebugStr(const Context&) const { return ""; }
};

template<typename Context, typename ValueT, typename E>
struct BaseExpr : public BaseBaseExpr<Context, ValueT> {
  virtual ValueT get(const Context&) const = 0;
  virtual std::string DebugStr(const Context&) const { return ""; }
};

template<typename Context, typename ValueT>
struct Literal final : public BaseExpr<Context, ValueT, Literal<Context, ValueT>> {
  Literal(const ValueT& val) : Literal("", val) {}  // NOLINT
  Literal(const std::string& debug_str, const ValueT& val) : val_(val), debug_str_(debug_str) {}
  ValueT get(const Context&) const { return val_; }

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
struct Custom final : public BaseExpr<Context, ValueT, Custom<Fn>> {
  Custom(Fn fn) : Custom(fn, "") {}  // NOLINT
  Custom(std::string debug_str, Fn fn) : fn_(std::move(fn)), debug_str_(std::move(debug_str)) {}
  ValueT get(const Context& context) const { return fn_(context); }

 private:
  Fn fn_;
  std::string debug_str_;
};

template<typename Fn>
inline Custom<Fn> make_custom(Fn fn) {
  return Custom<Fn>(std::forward<Fn>(fn));
}

template<typename Fn>
inline Custom<Fn> make_custom(const std::string& debug_str, Fn fn) {
  return Custom<Fn>(debug_str, std::forward<Fn>(fn));
}

template<typename Context, typename E>
using BaseBoolExpr = BaseExpr<Context, bool, E>;

template<typename Context>
struct TrueBoolExpr final : public BaseBoolExpr<Context, TrueBoolExpr<Context>> {
  bool get(const Context&) const override { return true; }
};

template<typename Context>
struct FalseBoolExpr final : public BaseBoolExpr<Context, FalseBoolExpr<Context>> {
  bool get(const Context&) const override { return false; }
};

template<typename Context, typename E1, typename E2>
struct AndBoolFunctor final : public BaseBoolExpr<Context, AndBoolFunctor<Context, E1, E2>> {
  AndBoolFunctor(E1 e1, E2 e2) : e1(std::move(e1)), e2(std::move(e2)) {}

  bool get(const Context& context) const override { return e1.get(context) & e2.get(context); }

 private:
  const E1 e1;
  const E2 e2;
};

template<typename Context, typename E1, typename E2>
struct OrBoolFunctor final : public BaseBoolExpr<Context, OrBoolFunctor<Context, E1, E2>> {
  OrBoolFunctor(E1 e1, E2 e2) : e1(std::move(e1)), e2(std::move(e2)) {}

  bool get(const Context& context) const override { return e1.get(context) & e2.get(context); }

 private:
  const E1 e1;
  const E2 e2;
};

template<typename Context, typename E>
struct NotBoolFunctor final : public BaseBoolExpr<Context, NotBoolFunctor<Context, E>> {
  explicit NotBoolFunctor(E e1) : e1(std::move(e1)) {}

  bool get(const Context& context) const override { return !e1.get(context); }

 private:
  const E e1;
};

template<typename Context, typename E1, typename E2>
struct EqualBoolFunctor final : public BaseBoolExpr<Context, EqualBoolFunctor<Context, E1, E2>> {
  EqualBoolFunctor(E1 e1, E2 e2) : e1(std::move(e1)), e2(std::move(e2)) {}

  bool get(const Context& context) const override { return e1.get(context) == e2.get(context); }

 private:
  const E1 e1;
  const E2 e2;
};

template<typename Context, typename E1, typename E2>
AndBoolFunctor<Context, E1, E2> operator&(BaseBoolExpr<Context, E1> const& u,
                                          BaseBoolExpr<Context, E2> const& v) {
  return AndBoolFunctor<Context, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
}

template<typename Context, typename E>
NotBoolFunctor<Context, E> operator~(BaseBoolExpr<Context, E> const& u) {
  return NotBoolFunctor<Context, E>(*static_cast<const E*>(&u));
}

template<typename Context, typename E1, typename E2>
OrBoolFunctor<Context, E1, E2> operator|(BaseBoolExpr<Context, E1> const& u,
                                         BaseBoolExpr<Context, E2> const& v) {
  return OrBoolFunctor<Context, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
}

template<typename Context, typename ValueT, typename E1, typename E2>
EqualBoolFunctor<Context, E1, E2> operator==(BaseExpr<Context, ValueT, E1> const& u,
                                             BaseExpr<Context, ValueT, E2> const& v) {
  return EqualBoolFunctor<Context, E1, E2>(*static_cast<const E1*>(&u),
                                           *static_cast<const E2*>(&v));
}

template<typename Context, typename ValueT, typename E1>
EqualBoolFunctor<Context, E1, Literal<Context, ValueT>> operator==(
    BaseExpr<Context, ValueT, E1> const& u, ValueT const& v) {
  return EqualBoolFunctor<Context, E1, Literal<Context, ValueT>>(*static_cast<const E1*>(&u),
                                                                 Literal<Context, ValueT>(v));
}

template<typename Context, typename E1>
EqualBoolFunctor<Context, E1, Literal<Context, std::string>> operator==(
    BaseExpr<Context, std::string, E1> const& u, const char* v) {
  return EqualBoolFunctor<Context, E1, Literal<Context, std::string>>(
      *static_cast<const E1*>(&u), Literal<Context, std::string>(v));
}

}  // namespace hob

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
