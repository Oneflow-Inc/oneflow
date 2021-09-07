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
#include "oneflow/core/common/error.cfg.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
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

  T value() const { return value_; }
  T& value() { return value_; }

  bool has_value() const { return init_; }

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

  T value() const { return *value_; }

  bool has_value() const { return value_; }

  void reset() { value_ = nullptr; }

 private:
  storage_type value_;
};

struct InPlaceConstruct {};

template<typename T>
class OptionalBase<
    T, typename std::enable_if<!IsScalarType<T>::value && !std::is_reference<T>::value>::type> {
 public:
  using value_type = T;
  using storage_type = std::shared_ptr<T>;

  OptionalBase() : value_(nullptr){};
  ~OptionalBase() = default;

  template<typename... Args>
  explicit OptionalBase(InPlaceConstruct, Args&&... args)
      : value_(std::make_shared<T>(std::forward<Args>(args)...)) {}

  explicit OptionalBase(const T& value) : value_(std::make_shared<T>(value)) {}
  explicit OptionalBase(T&& value) : value_(std::make_shared<T>(std::move(value))) {}

  explicit OptionalBase(const std::shared_ptr<T>& value) : value_(value) {}
  explicit OptionalBase(std::shared_ptr<T>&& value) : value_(std::move(value)) {}

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
  OptionalBase& operator=(const OptionalBase& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  OptionalBase& operator=(OptionalBase&& rhs) noexcept {
    value_ = std::move(rhs.value_);
    return *this;
  }

  const storage_type& value() const { return value_; }

  bool has_value() const { return bool(value_); }

  void reset() { value_.reset(); }

 private:
  std::shared_ptr<T> value_;
};

}  // namespace internal

using internal::InPlaceConstruct;

template<typename T>
class Optional final : private internal::OptionalBase<T> {
 private:
  using base = internal::OptionalBase<T>;

 public:
  using value_type = typename base::value_type;
  using storage_type = typename base::storage_type;

  using const_return_type = decltype(std::declval<const base>().value());
  using return_type = decltype(std::declval<base>().value());

  Optional() = default;
  ~Optional() = default;

  template<typename... Args>
  explicit Optional(Args&&... val) : base(std::forward<Args>(val)...) {}

  Optional(const Optional&) = default;
  Optional(Optional&&) noexcept = default;

  Optional& operator=(const T& val) {
    return static_cast<Optional&>(static_cast<base&>(*this) = val);
  }

  Optional& operator=(T&& val) {
    return static_cast<Optional&>(static_cast<base&>(*this) = std::move(val));
  }

  Optional& operator=(const Optional& rhs) = default;

  Optional& operator=(Optional&& rhs) noexcept = default;

  template<typename U>
  const_return_type value_or(U&& default_) const {
    if (has_value()) {
      return base::value();
    } else {
      return std::forward<U>(default_);
    }
  }

  template<typename U>
  return_type value_or(U&& default_) {
    if (has_value()) {
      return base::value();
    } else {
      return std::forward<U>(default_);
    }
  }

  bool has_value() const { return base::has_value(); }
  explicit operator bool() const { return has_value(); }

  bool IsOk() const { return has_value(); }

  const_return_type Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return base::value();
  }

  return_type Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() { return base::value(); }

  std::shared_ptr<cfg::ErrorProto> error() const {
    auto error = std::make_shared<cfg::ErrorProto>();
    error->mutable_value_not_found_error();
    return error;
  }

  Maybe<T> to_maybe() const {
    if (IsOk()) {
      return base::value();
    } else {
      return error();
    }
  }

  void reset() { base::reset(); }
};

template<typename T>
auto MakeOptional(T&& value)
    -> Optional<typename std::remove_cv<typename std::remove_reference<T>::type>::type> {
  return Optional<typename std::remove_cv<typename std::remove_reference<T>::type>::type>(
      std::forward<T>(value));
}

template<typename T, typename... Args>
auto ConstructOptional(Args&&... args) -> Optional<T> {
  return Optional<T>(InPlaceConstruct{}, std::forward<Args>(args)...);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPTIONAL_H_
