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

#include "oneflow/core/common/to_string.h"

namespace oneflow {

namespace hob {

template<typename T>
class BoolFunctor {
 public:
  virtual ~BoolFunctor() {}
  virtual bool operator()(const T& ctx) const = 0;
  virtual std::string DebugStr(const T& ctx, bool display_result = true) const = 0;

 protected:
  BoolFunctor() = default;
};

template<typename T>
class BoolFunctorPtr final {
 public:
  BoolFunctorPtr() = default;
  BoolFunctorPtr(const BoolFunctorPtr&) = default;
  BoolFunctorPtr(BoolFunctorPtr&&) = default;
  ~BoolFunctorPtr(){};
  BoolFunctorPtr(const std::shared_ptr<const BoolFunctor<T>>& ptr) : ptr_(ptr) {}

  BoolFunctorPtr operator&(const BoolFunctorPtr& ptr) const;
  BoolFunctorPtr operator|(const BoolFunctorPtr& ptr) const;
  BoolFunctorPtr operator~() const;

  BoolFunctorPtr& operator=(BoolFunctorPtr& ptr) {
    this->ptr_ = ptr.ptr_;
    return *this;
  }

  bool operator()(const T& ctx) const { return (*this->ptr_)(ctx); }

  std::string DebugStr(const T& ctx, bool display_result = true) const {
    return this->ptr_->DebugStr(ctx, display_result);
  }

 private:
  std::shared_ptr<const BoolFunctor<T>> ptr_;
};

template<typename T>
class AndBoolFunctor final : public BoolFunctor<T> {
 public:
  AndBoolFunctor() = delete;
  AndBoolFunctor(const BoolFunctorPtr<T> lhs, const BoolFunctorPtr<T> rhs) : lhs_(lhs), rhs_(rhs) {}
  ~AndBoolFunctor() {}

  bool operator()(const T& ctx) const override { return lhs_(ctx) && rhs_(ctx); }

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::string l_str = lhs_.DebugStr(ctx, display_result);
    display_result = display_result && lhs_(ctx);
    std::string r_str = rhs_.DebugStr(ctx, display_result);
    std::ostringstream string_stream;
    string_stream << "(" << l_str << " and " << r_str << ")";
    return string_stream.str();
  }

 private:
  const BoolFunctorPtr<T> lhs_;
  const BoolFunctorPtr<T> rhs_;
};

template<typename T>
class OrBoolFunctor final : public BoolFunctor<T> {
 public:
  OrBoolFunctor() = delete;
  OrBoolFunctor(const BoolFunctorPtr<T> lhs, const BoolFunctorPtr<T> rhs) : lhs_(lhs), rhs_(rhs) {}
  ~OrBoolFunctor() {}

  bool operator()(const T& ctx) const override { return lhs_(ctx) || rhs_(ctx); }

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::string l_str = lhs_.DebugStr(ctx, display_result);
    display_result = display_result && (!lhs_(ctx));
    std::string r_str = rhs_.DebugStr(ctx, display_result);
    std::ostringstream string_stream;
    string_stream << "(" << l_str << " or " << r_str << ")";
    return string_stream.str();
  }

 private:
  const BoolFunctorPtr<T> lhs_;
  const BoolFunctorPtr<T> rhs_;
};

template<typename T>
class NotBoolFunctor final : public BoolFunctor<T> {
 public:
  NotBoolFunctor() = delete;
  NotBoolFunctor(const BoolFunctorPtr<T> hs) : hs_(hs) {}
  ~NotBoolFunctor() {}

  bool operator()(const T& ctx) const override { return !hs_(ctx); }

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::ostringstream string_stream;
    string_stream << "("
                  << "not " << hs_.DebugStr(ctx, display_result) << ")";
    return string_stream.str();
  }

 private:
  const BoolFunctorPtr<T> hs_;
};

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator&(const BoolFunctorPtr& ptr) const {
  std::shared_ptr<const BoolFunctor<T>> and_ptr =
      std::make_shared<const AndBoolFunctor<T>>(this->ptr_, ptr.ptr_);
  return BoolFunctorPtr<T>(and_ptr);
}

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator|(const BoolFunctorPtr& ptr) const {
  std::shared_ptr<const BoolFunctor<T>> or_ptr =
      std::make_shared<const OrBoolFunctor<T>>(this->ptr_, ptr.ptr_);
  return BoolFunctorPtr<T>(or_ptr);
}

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator~() const {
  std::shared_ptr<const BoolFunctor<T>> not_ptr =
      std::make_shared<const NotBoolFunctor<T>>(this->ptr_);
  return BoolFunctorPtr<T>(not_ptr);
}

template<typename T>
class HighOrderBoolFunctor final : public hob::BoolFunctor<T> {
 public:
  HighOrderBoolFunctor() = delete;
  HighOrderBoolFunctor(const std::string& debug_str, const std::function<bool(const T&)>& bool_fn)
      : debug_str_(debug_str), bool_fn_(bool_fn) {}
  ~HighOrderBoolFunctor() {}

  bool operator()(const T& ctx) const override { return bool_fn_(ctx); }

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::ostringstream string_stream;
    string_stream << "(" << debug_str_;
    if (display_result) {
      std::string boolResult = bool_fn_(ctx) ? "True" : "False";
      string_stream << " [" << boolResult << "]";
    }
    string_stream << ")";
    return string_stream.str();
  }

 private:
  std::string debug_str_;
  std::function<bool(const T&)> bool_fn_;
};

template<typename ContextT, typename T>
class HobContextGetter final {
 public:
  HobContextGetter(const T& const_value)
      : debug_str_(ToString(const_value)),
        context_getter_([const_value](const ContextT&) { return const_value; }) {}
  HobContextGetter(const std::string& debug_str,
                   const std::function<T(const ContextT&)>& context_getter)
      : debug_str_(debug_str), context_getter_(context_getter) {}

#define GENERATE_OVERLOAD_OPERATOR_FUNC(op)                                        \
  BoolFunctorPtr<ContextT> operator op(const HobContextGetter& other) const {      \
    std::ostringstream string_stream;                                              \
    string_stream << debug_str_ << " " << #op << " " << other.debug_str_;          \
    std::function<T(const ContextT&)> l_fn = this->context_getter_;                \
    std::function<T(const ContextT&)> r_fn = other.context_getter_;                \
    std::shared_ptr<const BoolFunctor<ContextT>> krbf_ptr =                        \
        std::make_shared<const HighOrderBoolFunctor<ContextT>>(                    \
            string_stream.str(),                                                   \
            [l_fn, r_fn](const ContextT& ctx) { return l_fn(ctx) op r_fn(ctx); }); \
    return krbf_ptr;                                                               \
  }
  GENERATE_OVERLOAD_OPERATOR_FUNC(==)
  GENERATE_OVERLOAD_OPERATOR_FUNC(!=)
  GENERATE_OVERLOAD_OPERATOR_FUNC(>=)
  GENERATE_OVERLOAD_OPERATOR_FUNC(<=)
  GENERATE_OVERLOAD_OPERATOR_FUNC(>)
  GENERATE_OVERLOAD_OPERATOR_FUNC(<)
#undef GENERATE_OVERLOAD_OPERATOR_FUNC

 private:
  std::string debug_str_;
  std::function<T(const ContextT&)> context_getter_;
};

}  // namespace hob

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
