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

#ifndef ONEFLOW_CORE_COMMON_OPTIONAL_H_
#define ONEFLOW_CORE_COMMON_OPTIONAL_H_

#include <memory>
#include <type_traits>
#include <utility>
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/just.h"

namespace oneflow {

struct InPlaceConstructType {
  explicit InPlaceConstructType() = default;
};
constexpr InPlaceConstructType InPlaceConstruct{};

struct NullOptType {
  explicit constexpr NullOptType(int) {}
};
constexpr NullOptType NullOpt{0};

namespace internal {

template<typename T, typename U = void>
class OptionalBase;

template<typename T>
class OptionalBase<T, typename std::enable_if<IsScalarType<T>::value>::type> {
 public:
  using value_type = T;
  using storage_type = T;

  OptionalBase() : init_(false), value_() {}
  ~OptionalBase() = default;

  explicit OptionalBase(const T& value) : init_(true), value_(value) {}
  explicit OptionalBase(T&& value) : init_(true), value_(std::move(value)) {}

  OptionalBase(const OptionalBase& base) : init_(base.init_), value_(base.value_) {}
  OptionalBase(OptionalBase&& base) noexcept : init_(base.init_), value_(std::move(base.value_)) {}

  OptionalBase& operator=(const T& value) {
    value_ = value;
    init_ = true;

    return *this;
  }
  OptionalBase& operator=(T&& value) {
    value_ = std::move(value);
    init_ = true;

    return *this;
  }
  OptionalBase& operator=(const OptionalBase& rhs) {
    value_ = rhs.value_;
    init_ = rhs.init_;

    return *this;
  }
  OptionalBase& operator=(OptionalBase&& rhs) noexcept {
    value_ = std::move(rhs.value_);
    init_ = rhs.init_;

    return *this;
  }

  T value() const& { return value_; }  // `T value() &&` goes here
  T& value() & { return value_; }

  bool has_value() const { return init_; }

  T value_or(const T& other) const {
    if (has_value()) {
      return value();
    } else {
      return other;
    }
  }

  void reset() { init_ = false; }

 private:
  bool init_;
  T value_;
};

template<typename T>
class OptionalBase<T, typename std::enable_if<std::is_reference<T>::value>::type> {
 public:
  using value_type = typename std::remove_reference<T>::type;
  using storage_type = value_type*;

  static_assert(std::is_lvalue_reference<T>::value, "rvalue reference is not supported here");

  OptionalBase() : value_(nullptr){};
  ~OptionalBase() = default;

  explicit OptionalBase(T value) : value_(&value) {}
  OptionalBase(const OptionalBase& base) : value_(base.value_) {}
  OptionalBase(OptionalBase&& base) noexcept : value_(base.value_) {}

  OptionalBase& operator=(T value) {
    value_ = &value;
    return *this;
  }
  OptionalBase& operator=(const OptionalBase& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  OptionalBase& operator=(OptionalBase&& rhs) noexcept {
    value_ = std::move(rhs.value_);
    return *this;
  }

  const value_type& value() const { return *value_; }
  T value() { return *value_; }

  bool has_value() const { return value_; }

  const value_type& value_or(const value_type& other) const {
    if (has_value()) {
      return value();
    } else {
      return other;
    }
  }

  void reset() { value_ = nullptr; }

 private:
  storage_type value_;
};

template<typename T>
class OptionalBase<
    T, typename std::enable_if<!IsScalarType<T>::value && !std::is_reference<T>::value>::type> {
 public:
  using value_type = T;
  using storage_type = std::shared_ptr<T>;

  OptionalBase() : value_(nullptr){};
  ~OptionalBase() = default;

  template<typename... Args>
  explicit OptionalBase(InPlaceConstructType, Args&&... args)
      : value_(std::make_shared<T>(std::forward<Args>(args)...)) {}

  explicit OptionalBase(const T& value) : value_(std::make_shared<T>(value)) {}
  explicit OptionalBase(T&& value) : value_(std::make_shared<T>(std::move(value))) {}

  explicit OptionalBase(const storage_type& value) : value_(value) {}
  explicit OptionalBase(storage_type&& value) : value_(std::move(value)) {}

  OptionalBase(const OptionalBase&) = default;
  OptionalBase(OptionalBase&&) noexcept = default;

  OptionalBase& operator=(const T& value) {
    if (value_) {
      *value_ = value;
    } else {
      value_ = std::make_shared<T>(value);
    }
    return *this;
  }
  OptionalBase& operator=(T&& value) {
    if (value_) {
      *value_ = std::move(value);
    } else {
      value_ = std::make_shared<T>(std::move(value));
    }
    return *this;
  }

  OptionalBase& operator=(const storage_type& value) {
    value_ = value;
    return *this;
  }
  OptionalBase& operator=(storage_type&& value) {
    value_ = std::move(value);
    return *this;
  }

  OptionalBase& operator=(const OptionalBase& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  OptionalBase& operator=(OptionalBase&& rhs) noexcept {
    value_ = std::move(rhs.value_);
    return *this;
  }

  const storage_type& value() const& { return value_; }
  storage_type& value() & { return value_; }

  storage_type&& value() && { return std::move(value_); }

  bool has_value() const { return bool(value_); }

  const storage_type& value_or(const storage_type& other) const& {
    if (has_value()) {
      return value_;
    } else {
      return other;
    }
  }

  storage_type value_or(const storage_type& other) && {
    if (has_value()) {
      return std::move(value_);
    } else {
      return other;
    }
  }

  storage_type value_or(storage_type&& other) const& {
    if (has_value()) {
      return value_;
    } else {
      return std::move(other);
    }
  }

  storage_type value_or(storage_type&& other) && {
    if (has_value()) {
      return std::move(value_);
    } else {
      return std::move(other);
    }
  }

  // we introduce a dependent name `U` to delay the instantiation,
  // so only the default parameter of `U` is allowed
  template<typename U = value_type>
  typename std::enable_if<!std::is_abstract<U>::value, const U&>::type value_or(
      const value_type& other) const& {
    static_assert(std::is_same<U, value_type>::value, "expected default U");

    if (has_value()) {
      return *value_;
    } else {
      return other;
    }
  }

  template<typename U = value_type>
  typename std::enable_if<!std::is_abstract<U>::value, U>::type value_or(
      const value_type& other) && {
    static_assert(std::is_same<U, value_type>::value, "expected default U");

    if (has_value()) {
      return std::move(*value_);
    } else {
      return other;
    }
  }

  template<typename U = value_type>
  typename std::enable_if<!std::is_abstract<U>::value, U>::type value_or(
      value_type&& other) const& {
    static_assert(std::is_same<U, value_type>::value, "expected default U");

    if (has_value()) {
      return *value_;
    } else {
      return std::move(other);
    }
  }

  template<typename U = value_type>
  typename std::enable_if<!std::is_abstract<U>::value, U>::type value_or(value_type&& other) && {
    static_assert(std::is_same<U, value_type>::value, "expected default U");

    if (has_value()) {
      return std::move(*value_);
    } else {
      return std::move(other);
    }
  }

  void reset() { value_.reset(); }

 private:
  storage_type value_;
};

template<typename T>
struct IsOptional : std::false_type {};

template<typename T>
struct IsOptional<Optional<T>> : std::true_type {};

struct monadic_operations {
  template<typename T, typename F>
  static auto map(T&& opt, F&& f)
      -> Optional<decltype(std::forward<F>(f)(std::forward<T>(opt).value()))> {
    if (opt.has_value()) { return std::forward<F>(f)(std::forward<T>(opt).value()); }

    return NullOpt;
  }

  template<typename T, typename F,
           typename U = std::decay_t<decltype(std::declval<F>()(std::declval<T>().value()))>>
  static auto bind(T&& opt, F&& f) -> std::enable_if_t<IsOptional<U>::value, U> {
    if (opt.has_value()) { return std::forward<F>(f)(std::forward<T>(opt).value()); }

    return NullOpt;
  }

  template<typename T, typename F,
           std::enable_if_t<std::is_same<decltype(std::declval<F>()()), void>::value, int> = 0>
  static auto or_else(T&& opt, F&& f) -> std::decay_t<T> {
    if (!opt.has_value()) {
      std::forward<F>(f)();
      return NullOpt;
    }

    return std::forward<T>(opt);
  }

  template<typename T, typename F,
           std::enable_if_t<
               std::is_convertible<decltype(std::declval<F>()()), std::decay_t<T>>::value, int> = 0>
  static auto or_else(T&& opt, F&& f) -> std::decay_t<T> {
    if (!opt.has_value()) { return std::forward<F>(f)(); }

    return std::forward<T>(opt);
  }
};

}  // namespace internal

template<typename T>
class Optional final : private internal::OptionalBase<T> {
 private:
  using base = internal::OptionalBase<T>;
  using move_value_type = decltype(std::declval<base>().value());

 public:
  using value_type = typename base::value_type;
  using storage_type = typename base::storage_type;

  explicit Optional() = default;
  ~Optional() = default;

  Optional(NullOptType)  // NOLINT(google-explicit-constructor)
      : base() {}

  template<
      typename Arg1, typename... ArgN,
      typename std::enable_if<!(sizeof...(ArgN) == 0
                                && std::is_same<Optional, typename std::decay<Arg1>::type>::value),
                              int>::type = 0>
  Optional(Arg1&& v1, ArgN&&... vn)  // NOLINT(google-explicit-constructor)
      : base(std::forward<Arg1>(v1), std::forward<ArgN>(vn)...) {}

  Optional(const Optional&) = default;
  Optional(Optional&&) noexcept = default;

  template<typename U,
           typename std::enable_if<!std::is_same<Optional, typename std::decay<U>::type>::value,
                                   int>::type = 0>
  Optional& operator=(U&& val) {
    return static_cast<Optional&>(static_cast<base&>(*this) = std::forward<U>(val));
  }

  Optional& operator=(const Optional& rhs) = default;
  Optional& operator=(Optional&& rhs) noexcept = default;

  template<typename U>
  decltype(auto) value_or(U&& other) const& {
    return base::value_or(std::forward<U>(other));
  }

  template<typename U>
  decltype(auto) value_or(U&& other) && {
    return std::move(*this).base::value_or(std::forward<U>(other));
  }

  bool has_value() const { return base::has_value(); }
  explicit operator bool() const { return has_value(); }

  // generate a temporary object to allow `const auto& x = optval().value()` where `optval()` is a
  // function call which returns a temporary Optional
  auto Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() && -> std::conditional_t<
      std::is_rvalue_reference<move_value_type>::value, std::remove_reference_t<move_value_type>,
      move_value_type> {
    return std::move(*this).base::value();
  }

  friend internal::monadic_operations;

  template<typename F>
  auto map(F&& f) const& {
    return internal::monadic_operations::map(*this, std::forward<F>(f));
  }

  template<typename F>
  auto map(F&& f) && {
    return internal::monadic_operations::map(std::move(*this), std::forward<F>(f));
  }

  template<typename F>
  auto bind(F&& f) const& {
    return internal::monadic_operations::bind(*this, std::forward<F>(f));
  }

  template<typename F>
  auto bind(F&& f) && {
    return internal::monadic_operations::bind(std::move(*this), std::forward<F>(f));
  }

  template<typename F>
  auto or_else(F&& f) const& {
    return internal::monadic_operations::or_else(*this, std::forward<F>(f));
  }

  template<typename F>
  auto or_else(F&& f) && {
    return internal::monadic_operations::or_else(std::move(*this), std::forward<F>(f));
  }

  bool operator==(const Optional& other) const {
    if (has_value()) {
      if (other.has_value()) {
        return base::value() == other.base::value();
      } else {
        return false;
      }
    } else {
      return !other.has_value();
    }
  }

  bool operator!=(const Optional& other) const { return !operator==(other); }

  void reset() { base::reset(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPTIONAL_H_
