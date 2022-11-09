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

#ifndef ONEFLOW_MAYBE_JUST_H_
#define ONEFLOW_MAYBE_JUST_H_

#include <type_traits>
#include <utility>

#include "oneflow/maybe/error.h"
#include "oneflow/maybe/type_traits.h"

namespace oneflow {

namespace maybe {

template<typename T, typename E>
struct Maybe;

template<typename T>
struct IsMaybe : std::false_type {};

template<typename T, typename E>
struct IsMaybe<Maybe<T, E>> : std::true_type {};

template<typename T>
struct Optional;

template<typename T>
struct IsOptional : std::false_type {};

template<typename T>
struct IsOptional<Optional<T>> : std::true_type {};

// user should provide which error will be returned while an optional has no value
// and is used in JUST or CHECK_JUST;
// if not provided, then JUST(_MSG) and CHECK_JUST(_MSG) cannot be used for Optional
// i.e. ```c++
// template <typename T> struct JustConfig<Optional<T>> {
//   static SomeError ValueNotFoundError(auto&&) { ... }
// };
// ```
// or some other optional types, i.e. std::shared_ptr ```c++
// template <typename T> struct JustConfig<std::shared_ptr<T>> {
//   // define which error will be returned while it is empty
//   static SomeError ValueNotFoundError(auto&&) { ... }
//   // define how to get the underlying value
//   static decltype(auto) Value(auto&&) { ... }
// };
// ```
template<typename T>
struct JustTraits;

namespace details {

struct JustPrivateScope {
  template<typename T>
  static decltype(auto) Value(T&& v) {
    return std::forward<T>(v).Value();
  }

  template<typename T, std::enable_if_t<IsMaybe<RemoveCVRef<T>>::value, int> = 0>
  static decltype(auto) StackedError(T&& v) {
    return std::forward<T>(v).StackedError();
  }

  template<typename T, std::enable_if_t<!IsMaybe<RemoveCVRef<T>>::value, int> = 0>
  static decltype(auto) StackedError(T&& v) {
    return JustTraits<RemoveCVRef<T>>::ValueNotFoundError(std::forward<T>(v));
  }
};

template<typename T>
typename std::remove_const<typename std::remove_reference<T>::type>::type&& RemoveRValConst(
    T&& v) noexcept {
  static_assert(std::is_rvalue_reference<T&&>::value, "rvalue is expected here");
  return const_cast<typename std::remove_const<typename std::remove_reference<T>::type>::type&&>(v);
}

template<typename T, typename... Args>
decltype(auto) JustPushStackAndReturn(T&& v, Args&&... args) {
  StackedErrorTraits<RemoveCVRef<T>>::PushStack(std::forward<T>(v), std::forward<Args>(args)...);
  return std::forward<T>(v);
}

template<typename T, typename... Args>
[[noreturn]] void JustPushStackAndAbort(T&& v, Args&&... args) {
  using Traits = StackedErrorTraits<RemoveCVRef<T>>;

  Traits::PushStack(std::forward<T>(v), std::forward<Args>(args)...);
  Traits::Abort(std::forward<T>(v));
}

template<typename T, std::enable_if_t<IsMaybe<T>::value || IsOptional<T>::value, int> = 0>
auto JustGetValue(T&& v) -> RemoveRValRef<decltype(JustPrivateScope::Value(std::forward<T>(v)))> {
  return JustPrivateScope::Value(std::forward<T>(v));
}

template<typename T, std::enable_if_t<!IsMaybe<T>::value && !IsOptional<T>::value, int> = 0>
auto JustGetValue(T&& v)
    -> RemoveRValRef<decltype(JustTraits<RemoveCVRef<T>>::Value(std::forward<T>(v)))> {
  return JustTraits<RemoveCVRef<T>>::Value(std::forward<T>(v));
}

}  // namespace details

}  // namespace maybe

}  // namespace oneflow

// macros begin

#define JUST_STACK_CHECK_I(...) __VA_ARGS__

#define JUST_TO_STR_I(...) #__VA_ARGS__

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

#define JUST(...)                                                                       \
  ::oneflow::maybe::details::JustGetValue(::oneflow::maybe::details::RemoveRValConst(({ \
    auto&& _just_value_to_check_ = JUST_STACK_CHECK_I(__VA_ARGS__);                     \
    if (OF_MAYBE_EXPECT_FALSE(!_just_value_to_check_)) {                                \
      return ::oneflow::maybe::details::JustPushStackAndReturn(                         \
          ::oneflow::maybe::details::JustPrivateScope::StackedError(                    \
              std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_)),    \
          __FILE__, __LINE__, __PRETTY_FUNCTION__, JUST_TO_STR_I(__VA_ARGS__));         \
    }                                                                                   \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);               \
  })))

#define CHECK_JUST(...)                                                              \
  ::oneflow::maybe::details::JustGetValue([&](const auto& _just_function_name_) {    \
    auto&& _just_value_to_check_ = JUST_STACK_CHECK_I(__VA_ARGS__);                  \
    if (OF_MAYBE_EXPECT_FALSE(!_just_value_to_check_)) {                             \
      ::oneflow::maybe::details::JustPushStackAndAbort(                              \
          ::oneflow::maybe::details::JustPrivateScope::StackedError(                 \
              std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_)), \
          __FILE__, __LINE__, _just_function_name_, JUST_TO_STR_I(__VA_ARGS__));     \
    }                                                                                \
    return std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);     \
  }(__PRETTY_FUNCTION__))

#define JUST_MSG(_just_expr_, ...)                                                           \
  ::oneflow::maybe::details::JustGetValue(::oneflow::maybe::details::RemoveRValConst(({      \
    auto&& _just_value_to_check_ = (_just_expr_);                                            \
    if (OF_MAYBE_EXPECT_FALSE(!_just_value_to_check_)) {                                     \
      return ::oneflow::maybe::details::JustPushStackAndReturn(                              \
          ::oneflow::maybe::details::JustPrivateScope::StackedError(                         \
              std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_)),         \
          __FILE__, __LINE__, __PRETTY_FUNCTION__, JUST_TO_STR_I(_just_expr_), __VA_ARGS__); \
    }                                                                                        \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);                    \
  })))

#define CHECK_JUST_MSG(_just_expr_, ...)                                                      \
  ::oneflow::maybe::details::JustGetValue([&](const auto& _just_function_name_) {             \
    auto&& _just_value_to_check_ = (_just_expr_);                                             \
    if (OF_MAYBE_EXPECT_FALSE(!_just_value_to_check_)) {                                      \
      ::oneflow::maybe::details::JustPushStackAndAbort(                                       \
          ::oneflow::maybe::details::JustPrivateScope::StackedError(                          \
              std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_)),          \
          __FILE__, __LINE__, _just_function_name_, JUST_TO_STR_I(_just_expr_), __VA_ARGS__); \
    }                                                                                         \
    return std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);              \
  }(__PRETTY_FUNCTION__))

#define OPT_JUST(...)                                                                   \
  ::oneflow::maybe::details::JustGetValue(::oneflow::maybe::details::RemoveRValConst(({ \
    auto&& _just_value_to_check_ = JUST_STACK_CHECK_I(__VA_ARGS__);                     \
    if (OF_MAYBE_EXPECT_FALSE(!_just_value_to_check_)) { return NullOpt; }              \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);               \
  })))

#else
#error "statement expression is not supported, please implement try-catch version of JUST"
#endif  // defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

// macros end

#endif  // ONEFLOW_MAYBE_JUST_H_
