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

#ifndef ONEFLOW_MAYBE_VARIANT_H_
#define ONEFLOW_MAYBE_VARIANT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <functional>
#include <iostream>

#include "oneflow/maybe/utility.h"
#include "oneflow/maybe/type_traits.h"

namespace oneflow {

namespace maybe {

template<typename... Ts>
struct Variant;

namespace details {

// there are generally two ways to implement visit (like std::visit in c++17)
// 1. O(N) or O(log N), to iterate for all types or do a binary search on type index recursively
// 2. O(1), to store an static (storage duration) array of function pointers for every (Variant, F)
// where N = Variant<T...>::Num, and normally within the range [2, 5]
// the 2nd method is required in std::visit(f, x...) while sizeof...(x) == 1
// but weakness of the 2nd method is that compilers usually cannot efficiently optimize these function pointers
// (compared to trivial recursion, which is easy to do optimization, and also friendly to CPU cache)
// here we implement visit via the first method
template<typename R, typename T, typename F, typename V>
R VisitImpl(F&& f, V&& v) {
  // assume v.template Is<T>() now
  return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Get<T>()));
}

template<typename R, typename T1, typename... Tn, typename F, typename V,
         std::enable_if_t<(sizeof...(Tn) > 0), int> = 0>
R VisitImpl(F&& f, V&& v) {
  if (v.template Is<T1>()) {
    return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Get<T1>()));
  }

  return VisitImpl<R, Tn...>(std::forward<F>(f), std::forward<V>(v));
}

template<typename R, typename F, typename... Ts>
struct VisitResultT {
  using type = R;
};

template<typename F, typename... Ts>
struct VisitResultT<DefaultArgument, F, Ts...> {
  using type = std::common_type_t<decltype(std::declval<F>()(std::declval<Ts>()))...>;
};

template<typename R, typename F, typename... Ts>
using VisitResult = typename VisitResultT<R, F, Ts...>::type;

}  // namespace details

// preconditions: template type arguments must be no less than 2 different type 
// and without reference and cv qualifiers
// this Variant DO NOT guarantee exception safty
template<typename... Ts>
struct Variant {  // NOLINT(cppcoreguidelines-pro-type-member-init)
 public:
  static_assert(sizeof...(Ts) > 1, "expected more than two types");
  static_assert(Conj<NegT<std::is_reference<Ts>>...>, "reference types are not allowed here");
  static_assert(Conj<NegT<DisjT<std::is_const<Ts>, std::is_volatile<Ts>>>...>,
                "cv qualifiers are not allowed here");
  static_assert(IsDifferentTypes<Ts...>, "expected all of different types");

  static constexpr std::size_t Num = sizeof...(Ts);

  template<typename T>
  static constexpr auto IndexOfType = IndexGet<T, Ts...>;

  template<typename T>
  static constexpr bool HasType = TypeIn<T, Ts...>;

  template<std::size_t I>
  using TypeByIndex = TypeGet<I, Ts...>;

  template<typename T = TypeByIndex<0>,
           std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
  Variant() {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    construct<0>();
  }

  // unlike std::variant, we only accept exact types to avoid wrong construction
  template<typename T, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  Variant(T&& v) {  // NOLINT(cppcoreguidelines-pro-type-member-init, google-explicit-constructor)
    construct<RemoveCVRef<T>>(std::forward<T>(v));
  }

  template<typename T, typename... Args, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  explicit Variant(InPlaceTypeT<T>,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                   Args&&... args) {
    construct<RemoveCVRef<T>>(std::forward<Args>(args)...);
  }

  template<std::size_t I, typename... Args, std::enable_if_t<(I < Num), int> = 0>
  explicit Variant(InPlaceIndexT<I>,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                   Args&&... args) {
    construct<I>(std::forward<Args>(args)...);
  }

  template<typename R = DefaultArgument, typename F>
  auto visit(F&& f) & {
    using Result = details::VisitResult<R, F, Ts&...>;
    return details::VisitImpl<Result, Ts...>(std::forward<F>(f), *this);
  }

  template<typename R = DefaultArgument, typename F>
  auto visit(F&& f) && {
    using Result = details::VisitResult<R, F, Ts&&...>;
    return details::VisitImpl<Result, Ts...>(std::forward<F>(f), std::move(*this));
  }

  template<typename R = DefaultArgument, typename F>
  auto visit(F&& f) const& {
    using Result = details::VisitResult<R, F, const Ts&...>;
    return details::VisitImpl<Result, Ts...>(std::forward<F>(f), *this);
  }

  Variant(const Variant& v) {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    copyConstruct(v);
  }

  Variant(Variant&& v) noexcept {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    copyConstruct(std::move(v));
  }

  template<typename T, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  Variant& operator=(T&& v) {
    using Type = RemoveCVRef<T>;

    Emplace<Type>(std::forward<T>(v));

    return *this;
  }

  Variant& operator=(const Variant& v) {
    copy(v);
    return *this;
  }

  Variant& operator=(Variant&& v) noexcept {
    copy(std::move(v));
    return *this;
  }

  std::size_t Index() const { return index; }

  template<typename T>
  bool Is() const {
    return index == IndexOfType<T>;
  }

  // use std::launder while updating to c++17
  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T& Get() & {
    return *reinterpret_cast<T*>(storage);
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T&& Get() && {
    return std::move(*reinterpret_cast<T*>(storage));
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  const T& Get() const& {
    return *reinterpret_cast<const T*>(storage);
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>& Get() & {
    return *reinterpret_cast<TypeByIndex<I>*>(storage);
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>&& Get() && {
    return std::move(*reinterpret_cast<TypeByIndex<I>*>(storage));
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  const TypeByIndex<I>& Get() const& {
    return *reinterpret_cast<const TypeByIndex<I>*>(storage);
  }

  ~Variant() { destory(); }

  bool operator==(const Variant& v) const {
    if (index != v.index) return false;

    return v.visit([this](const auto& elem) { return elem == Get<RemoveCVRef<decltype(elem)>>(); });
  }

  bool operator!=(const Variant& v) const { return !operator==(v); }

  template<typename T, typename... Args>
  T& Emplace(Args&&... args) {
    if (Is<T>()) {
      return Get<T>() = T(std::forward<Args>(args)...);
    } else {
      destory();
      construct<T>(std::forward<Args>(args)...);
      return Get<T>();
    }
  }

  template<std::size_t I, typename... Args>
  decltype(auto) Emplace(Args&&... args) {
    return Emplace<TypeByIndex<I>>(std::forward<Args>(args)...);
  }

 private:
  static constexpr const std::size_t size = std::max({sizeof(Ts)...});

  alignas(Ts...) unsigned char storage[size];
  std::uint8_t index;

  template<typename T, typename... Args, std::enable_if_t<HasType<T>, int> = 0>
  void construct(Args&&... args) {
    new (storage) T(std::forward<Args>(args)...);
    index = IndexOfType<T>;
  }

  template<std::size_t I, typename... Args, std::enable_if_t<(I < Num), int> = 0>
  void construct(Args&&... args) {
    new (storage) TypeByIndex<I>(std::forward<Args>(args)...);
    index = I;
  }

  template<typename V>
  void copyConstruct(V&& v) {
    std::forward<V>(v).visit([this](auto&& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      new (storage) T(std::forward<decltype(elem)>(elem));
      index = IndexOfType<T>;
    });
  }

  template<typename V>
  void copy(V&& v) {
    std::forward<V>(v).visit([this](auto&& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      if (Is<T>()) {
        Get<T>() = std::forward<decltype(elem)>(elem);
      } else {
        destory();
        construct<T>(std::forward<decltype(elem)>(elem));
      }
    });
  }

  void destory() {
    visit([this](auto& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      Get<T>().~T();
    });
  }
};

template<typename... Ts>
using OptionalVariant = Variant<NullOptType, Ts...>;

}  // namespace maybe

}  // namespace oneflow

namespace std {

template<typename... Ts>
struct hash<oneflow::maybe::Variant<Ts...>> {
  size_t operator()(const oneflow::maybe::Variant<Ts...>& v) const noexcept {
    size_t seed = hash<size_t>()(v.Index());

    v.visit([&seed](const auto& x) {
      using type = oneflow::maybe::RemoveCVRef<decltype(x)>;
      oneflow::maybe::HashCombine<type>(seed, x);
    });

    return seed;
  }
};

}  // namespace std

#endif  // ONEFLOW_MAYBE_VARIANT_H_
