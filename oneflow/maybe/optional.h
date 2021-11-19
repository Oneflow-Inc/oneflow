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

#ifndef ONEFLOW_MAYBE_OPTIONAL_H_
#define ONEFLOW_MAYBE_OPTIONAL_H_

#include <type_traits>
#include <utility>

#include "oneflow/maybe/utility.h"
#include "oneflow/maybe/type_traits.h"

namespace oneflow {

namespace maybe {

template<typename T>
struct Optional;

namespace details {

// OptionalStorage is specialized for 2 cases:
// 1. for scalar types, we optimize all construction, destruction and value check
// 2. for reference types, we store a pointer to the referenced value
template<typename T, typename = void>
struct OptionalStorage {
 private:
  bool has;
  alignas(T) unsigned char value[sizeof(T)];

  using Type = std::remove_const_t<T>;

 public:
  void Init() { has = false; }

  T& Value() & { return *reinterpret_cast<T*>(value); }

  Type&& Value() && { return std::move(*const_cast<Type*>(reinterpret_cast<T*>(value))); }

  const T& Value() const& { return *reinterpret_cast<const T*>(value); }

  bool HasValue() const { return has; }

  void Reset() {
    if (has) {
      has = false;
      Value().~T();
    }
  }

  void Destory() {
    if (has) { Value().~T(); }
  }

  template<typename... Args>
  void Construct(Args&&... args) {
    new (value) Type{std::forward<Args>(args)...};
    has = true;
  }

  template<typename... Args, typename U = T, std::enable_if_t<!std::is_const<U>::value, int> = 0>
  T& Emplace(Args&&... args) {
    if (!has) {
      Construct(std::forward<Args>(args)...);
      return Value();
    } else {
      return Value() = Type(std::forward<Args>(args)...);
    }
  }

  template<typename... Args, typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
  T& Emplace(Args&&... args) {
    Destory();
    Construct(std::forward<Args>(args)...);
    return Value();
  }

  template<typename OS>
  void CopyConstruct(OS&& s) {
    has = s.has;

    if (has) { new (value) Type(std::forward<OS>(s).Value()); }
  }

  template<typename OS>
  void Copy(OS&& s) {
    if (s.has) {
      Emplace(std::forward<OS>(s).Value());
    } else {
      Reset();
    }
  }
};

template<typename T>
struct OptionalStorage<T, std::enable_if_t<std::is_scalar<T>::value>> {
 private:
  using Type = std::remove_const_t<T>;

  bool has;
  Type value;

 public:
  void Init() { has = false; }

  T& Value() & { return value; }

  Type&& Value() && { return std::move(const_cast<Type&>(value)); }

  const T& Value() const& { return value; }

  bool HasValue() const { return has; }

  void Reset() { has = false; }

  void Destory() {}

  template<typename U>
  void Construct(const U& v) {
    value = v;
    has = true;
  }

  template<typename U>
  T& Emplace(const U& v) {
    Construct(v);
    return Value();
  }

  void CopyConstruct(const OptionalStorage& s) {
    has = s.has;
    value = s.value;
  }

  void Copy(const OptionalStorage& s) { CopyConstruct(s); }
};

template<typename T>
struct OptionalStorage<T, std::enable_if_t<std::is_reference<T>::value>> {
  static_assert(std::is_lvalue_reference<T>::value, "rvalue reference is not allowed here");

  using Type = std::remove_reference_t<T>;

 private:
  Type* value;

 public:
  void Init() { value = nullptr; }

  T Value() { return *value; }

  const Type& Value() const { return *value; }

  bool HasValue() const { return value != nullptr; }

  void Reset() { value = nullptr; }

  void Destory() {}

  void Construct(T v) { value = &v; }

  T Emplace(T v) {
    Construct(v);
    return Value();
  }

  void CopyConstruct(const OptionalStorage& s) { value = s.value; }

  void Copy(const OptionalStorage& s) { CopyConstruct(s); }
};

template<typename T>
struct IsOptional : std::false_type {};

template<typename T>
struct IsOptional<Optional<T>> : std::true_type {};

struct OptionalPrivateScope {
  template<typename T>
  static decltype(auto) Value(T&& opt) {
    return std::forward<T>(opt).Value();
  }

  template<typename T, typename F>
  static auto Map(T&& opt, F&& f)
      -> Optional<decltype(std::forward<F>(f)(std::forward<T>(opt).Value()))> {
    if (opt.HasValue()) { return std::forward<F>(f)(std::forward<T>(opt).Value()); }

    return NullOpt;
  }

  template<typename T, typename F,
           typename U = std::decay_t<decltype(std::declval<F>()(std::declval<T>().Value()))>>
  static auto Bind(T&& opt, F&& f) -> std::enable_if_t<IsOptional<U>::value, U> {
    if (opt.HasValue()) { return std::forward<F>(f)(std::forward<T>(opt).Value()); }

    return NullOpt;
  }

  template<typename T, typename F,
           std::enable_if_t<std::is_same<decltype(std::declval<F>()()), void>::value, int> = 0>
  static auto OrElse(T&& opt, F&& f) -> std::decay_t<T> {
    if (!opt.HasValue()) {
      std::forward<F>(f)();
      return NullOpt;
    }

    return std::forward<T>(opt);
  }

  template<typename T, typename F,
           std::enable_if_t<
               std::is_convertible<decltype(std::declval<F>()()), std::decay_t<T>>::value, int> = 0>
  static auto OrElse(T&& opt, F&& f) -> std::decay_t<T> {
    if (!opt.HasValue()) { return std::forward<F>(f)(); }

    return std::forward<T>(opt);
  }
};

}  // namespace details

// unlike Variant, type arguments can be cv qualified or lvalue referenced
// this Optional DO NOT guarantee exception safty
template<typename T>
struct Optional {
 private:
  details::OptionalStorage<T> storage;

  using Type = std::remove_const_t<T>;

  decltype(auto) Value() & { return storage.Value(); }

  decltype(auto) Value() && { return std::move(storage).Value(); }

  decltype(auto) Value() const& { return storage.Value(); }

  // we DO NOT export Value method, then leave these methods accessable for the JUST macro
  friend struct details::OptionalPrivateScope;

 public:
  static_assert(!std::is_same<Type, NullOptType>::value, "NullOptType is not allowed in Optional");

  using ValueType = T;

  explicit Optional() { storage.Init(); };

  Optional(NullOptType) { storage.Init(); }  // NOLINT(google-explicit-constructor)

  Optional(const T& v) { storage.Construct(v); }  // NOLINT(google-explicit-constructor)

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  Optional(Type&& v) {  // NOLINT(google-explicit-constructor)
    storage.Construct(std::move(v));
  }

  Optional(const Optional& opt) { storage.CopyConstruct(opt.storage); }
  Optional(Optional&& opt) noexcept { storage.CopyConstruct(std::move(opt.storage)); }

  template<typename... Args>
  explicit Optional(InPlaceT, Args&&... args) {
    storage.Construct(std::forward<Args>(args)...);
  }

  ~Optional() { storage.Destory(); }

  Optional& operator=(NullOptType) {
    storage.Reset();
    return *this;
  }

  Optional& operator=(const T& v) {
    storage.Emplace(v);
    return *this;
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  Optional& operator=(Type&& v) {
    storage.Emplace(std::move(v));
    return *this;
  }

  template<typename... Args>
  decltype(auto) Emplace(Args&&... args) {
    return storage.Emplace(std::forward<Args>(args)...);
  }

  Optional& operator=(const Optional& opt) {
    storage.Copy(opt.storage);
    return *this;
  }

  Optional& operator=(Optional&& opt) noexcept {
    storage.Copy(std::move(opt.storage));
    return *this;
  }

  bool HasValue() const { return storage.HasValue(); }
  explicit operator bool() const { return HasValue(); }

  bool operator==(const Optional& opt) const {
    if (HasValue()) {
      if (opt.HasValue()) {
        return Value() == opt.Value();
      } else {
        return false;
      }
    } else {
      return !opt.HasValue();
    }
  }

  bool operator!=(const Optional& opt) const { return !operator==(opt); }

  bool operator<(const Optional& opt) const {
    if (HasValue()) {
      if (opt.HasValue()) {
        return Value() < opt.Value();
      } else {
        return false;
      }
    } else {
      return opt.HasValue();
    }
  }

  bool operator>=(const Optional& opt) const { return !operator<(opt); }

  bool operator>(const Optional& opt) const {
    if (HasValue()) {
      if (opt.HasValue()) {
        return Value() > opt.Value();
      } else {
        return true;
      }
    } else {
      return false;
    }
  }

  bool operator<=(const Optional& opt) const { return !operator>(opt); }

  friend bool operator==(const Optional& opt, NullOptType) { return !opt.HasValue(); }
  friend bool operator!=(const Optional& opt, NullOptType) { return opt.HasValue(); }
  friend bool operator==(NullOptType, const Optional& opt) { return !opt.HasValue(); }
  friend bool operator!=(NullOptType, const Optional& opt) { return opt.HasValue(); }

  friend bool operator<(const Optional& opt, NullOptType) { return false; }
  friend bool operator>(const Optional& opt, NullOptType) { return opt.HasValue(); }
  friend bool operator<=(const Optional& opt, NullOptType) { return !opt.HasValue(); }
  friend bool operator>=(const Optional& opt, NullOptType) { return true; }

  friend bool operator<(NullOptType, const Optional& opt) { return opt > NullOpt; }
  friend bool operator>(NullOptType, const Optional& opt) { return opt < NullOpt; }
  friend bool operator<=(NullOptType, const Optional& opt) { return opt >= NullOpt; }
  friend bool operator>=(NullOptType, const Optional& opt) { return opt <= NullOpt; }

  friend bool operator==(const Optional& opt, const T& v) {
    if (opt.HasValue()) {
      return opt.Value() == v;
    } else {
      return false;
    }
  }

  friend bool operator!=(const Optional& opt, const T& v) { return !(opt == v); }

  friend bool operator==(const T& v, const Optional& opt) { return opt == v; }

  friend bool operator!=(const T& v, const Optional& opt) { return !(opt == v); }

  friend bool operator<(const Optional& opt, const T& v) {
    if (opt.HasValue()) {
      return opt.Value() < v;
    } else {
      return true;
    }
  }

  friend bool operator>=(const Optional& opt, const T& v) { return !(opt < v); }

  friend bool operator>(const T& v, const Optional& opt) { return opt < v; }

  friend bool operator<=(const T& v, const Optional& opt) { return !(opt < v); }

  friend bool operator>(const Optional& opt, const T& v) {
    if (opt.HasValue()) {
      return opt.Value() > v;
    } else {
      return false;
    }
  }

  friend bool operator<=(const Optional& opt, const T& v) { return !(opt > v); }

  friend bool operator<(const T& v, const Optional& opt) { return opt > v; }

  friend bool operator>=(const T& v, const Optional& opt) { return !(opt > v); }

  decltype(auto) ValueOr(const T& v) const& {
    if (HasValue()) {
      return Value();
    } else {
      return v;
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  auto ValueOr(T&& v) const& {
    if (HasValue()) {
      return Value();
    } else {
      return std::move(v);
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  auto ValueOr(const T& v) && {
    if (HasValue()) {
      return std::move(*this).Value();
    } else {
      return v;
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  decltype(auto) ValueOr(T&& v) && {
    if (HasValue()) {
      return std::move(*this).Value();
    } else {
      return std::move(v);
    }
  }

  void Reset() { storage.Reset(); }

  template<typename F>
  auto Map(F&& f) const& {
    return details::OptionalPrivateScope::Map(*this, std::forward<F>(f));
  }

  template<typename F>
  auto Map(F&& f) && {
    return details::OptionalPrivateScope::Map(std::move(*this), std::forward<F>(f));
  }

  template<typename F>
  auto Bind(F&& f) const& {
    return details::OptionalPrivateScope::Bind(*this, std::forward<F>(f));
  }

  template<typename F>
  auto Bind(F&& f) && {
    return details::OptionalPrivateScope::Bind(std::move(*this), std::forward<F>(f));
  }

  template<typename F>
  auto OrElse(F&& f) const& {
    return details::OptionalPrivateScope::OrElse(*this, std::forward<F>(f));
  }

  template<typename F>
  auto OrElse(F&& f) && {
    return details::OptionalPrivateScope::OrElse(std::move(*this), std::forward<F>(f));
  }
};

}  // namespace maybe

}  // namespace oneflow

namespace std {

template<typename T>
struct hash<oneflow::maybe::Optional<T>> {
  size_t operator()(const oneflow::maybe::Optional<T>& v) const noexcept {
    if (v.HasValue()) {
      return hashImpl(oneflow::maybe::details::OptionalPrivateScope::Value(v));
    } else {
      return oneflow::maybe::NullOptHash;
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  static std::size_t hashImpl(const T& v) {
    return std::hash<std::remove_cv_t<T>>()(v);
  }

  template<typename U = T, std::enable_if_t<std::is_reference<U>::value, int> = 0>
  static std::size_t hashImpl(const std::remove_reference_t<T>& v) {
    return std::hash<const std::remove_reference_t<T>*>()(&v);
  }
};

}  // namespace std

#endif  // ONEFLOW_MAYBE_OPTIONAL_H_
