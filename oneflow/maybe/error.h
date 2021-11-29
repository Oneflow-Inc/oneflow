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

#ifndef ONEFLOW_MAYBE_ERROR_H_
#define ONEFLOW_MAYBE_ERROR_H_

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <iostream>

#include "utility.h"
#include "type_traits.h"
#include "string_view.h"

namespace oneflow {

namespace maybe {

namespace details {

template<typename D>
struct ErrorStackFromContainerBase {
 private:
  using Derived = D;

  auto& stack() { return static_cast<Derived*>(this)->getStack(); }

  const auto& stack() const { return static_cast<const Derived*>(this)->getStack(); }

 public:
  std::size_t StackSize() const { return stack().size(); }

  template<typename... Args>
  void PushStack(Args&&... args) {
    auto& s = stack();
    s.emplace(s.end(), std::forward<Args>(args)...);
  }

  template<typename T = Derived>
  const typename T::StackType::value_type& StackElem(std::size_t index) const {
    return stack()[index];
  }

  auto StackBegin() const { return stack().begin(); }
  auto StackEnd() const { return stack().end(); }
};

}  // namespace details

template<typename T>
struct StackedErrorTraits {
  StackedErrorTraits() = delete;

  using ErrorType = typename T::ErrorType;
  using StackEntryType = typename T::StackEntryType;

  template<
      typename U,
      std::enable_if_t<
          std::is_same<T, RemoveCVRef<U>>::value
              && std::is_same<ErrorType, RemoveCVRef<decltype(std::declval<U>().Error())>>::value,
          int> = 0>
  static decltype(auto) Error(U&& se) {
    return se.Error();
  }

  static std::size_t StackSize(const T& se) { return se.StackSize(); }

  static ConstRefExceptVoid<StackEntryType> StackElem(const T& se, std::size_t index) {
    return se.StackElem(index);
  }

  template<typename U, typename... Args,
           std::enable_if_t<std::is_same<T, RemoveCVRef<U>>::value, int> = 0>
  static void PushStack(U&& se, Args&&... args) {
    se.PushStack(std::forward<Args>(args)...);
  }

  template<typename U, std::enable_if_t<std::is_same<T, RemoveCVRef<U>>::value, int> = 0>
  static std::string Dump(U&& se) {
    return se.Dump();
  }

  template<typename U, std::enable_if_t<std::is_same<T, RemoveCVRef<U>>::value, int> = 0>
  [[noreturn]] static void Abort(U&& se) {
    se.Abort();
  }
};

template<typename T>
struct StackedErrorTraits<std::unique_ptr<T>> {
  StackedErrorTraits() = delete;

  using PointedTraits = StackedErrorTraits<T>;

  using ValueType = std::unique_ptr<T>;

  using ErrorType = typename PointedTraits::ErrorType;
  using StackEntryType = typename PointedTraits::StackEntryType;

  template<typename U, std::enable_if_t<std::is_same<ValueType, RemoveCVRef<U>>::value, int> = 0>
  static decltype(auto) Error(U&& se) {
    return PointedTraits::Error(*se);
  }

  static std::size_t StackSize(const ValueType& se) { return PointedTraits::StackSize(*se); }

  static ConstRefExceptVoid<StackEntryType> StackElem(const T& se, std::size_t index) {
    return PointedTraits::StackElem(*se, index);
  }

  template<typename U, typename... Args,
           std::enable_if_t<std::is_same<ValueType, RemoveCVRef<U>>::value, int> = 0>
  static void PushStack(U&& se, Args&&... args) {
    PointedTraits::PushStack(*se, std::forward<Args>(args)...);
  }

  template<typename U, std::enable_if_t<std::is_same<ValueType, RemoveCVRef<U>>::value, int> = 0>
  static std::string Dump(U&& se) {
    return PointedTraits::Dump(*se);
  }

  template<typename U, std::enable_if_t<std::is_same<ValueType, RemoveCVRef<U>>::value, int> = 0>
  [[noreturn]] static void Abort(U&& se) {
    PointedTraits::Abort(*se);
  }
};

// simple implementation for some customization points
namespace simple {

template<typename T>
struct MessageFormatTrait;

template<>
struct MessageFormatTrait<std::string> {
  template<typename Code, typename... Args>
  static std::string Format(Code&& code, Args&&... args) {
    if (sizeof...(args) > 0) {
      std::stringstream res;

      res << code << ": ";
      [[maybe_unused]] int dummy[] = {(res << args, 0)...};

      return res.str();
    } else {
      return code;
    }
  }
};

template<>
struct MessageFormatTrait<StringView> {
  template<typename Code>
  static StringView Format(Code&& code) {
    return code;
  }
};

template<typename Message, typename MessageFormatTraits = MessageFormatTrait<Message>>
struct ErrorStackEntry {
  StringView filename;
  std::size_t lineno;
  StringView function;
  Message message;

  template<typename... Args>
  ErrorStackEntry(StringView filename, std::size_t lineno, StringView function, Args&&... args)
      : filename(filename),
        lineno(lineno),
        function(function),
        message(MessageFormatTraits::Format(std::forward<Args>(args)...)) {}
};

template<typename E, typename M = std::string>
struct StackedError : details::ErrorStackFromContainerBase<StackedError<E, M>> {
 public:
  using ErrorType = E;
  using StackMessage = M;
  using StackEntryType = ErrorStackEntry<StackMessage>;
  using StackType = std::vector<StackEntryType>;
  using BaseType = details::ErrorStackFromContainerBase<StackedError<E, M>>;

  StackedError(ErrorType error)  // NOLINT(google-explicit-constructor)
      : error(std::move(error)) {}

  ErrorType& Error() { return error; }
  const ErrorType& Error() const { return error; }

  std::string Dump() {
    std::stringstream res;
    res << "error occurred: " << error << std::endl;
    for (const auto& elem : stack) {
      res << "from " << elem.function << " in " << elem.filename << ":" << elem.lineno << ": "
          << elem.message << std::endl;
    }

    return res.str();
  }

  [[noreturn]] void Abort() {
    std::cerr << "error occurred: " << error << std::endl;
    for (const auto& elem : stack) {
      std::cerr << "from " << elem.function << " in " << elem.filename << ":" << elem.lineno << ": "
                << elem.message << std::endl;
    }
    std::abort();
  }

 private:
  ErrorType error;
  StackType stack;

  StackType& getStack() { return stack; }

  const StackType& getStack() const { return stack; }

  friend BaseType;
};

template<typename E>
struct NoStackError {
  using ErrorType = E;
  using StackEntryType = void;

  NoStackError(ErrorType error)  // NOLINT(google-explicit-constructor)
      : error(std::move(error)) {}

  ErrorType& Error() { return error; }
  const ErrorType& Error() const { return error; }

  std::size_t StackSize() const { return 0; }

  void StackElem(std::size_t) const {}

  template<typename... Args>
  void PushStack(Args&&... args) {}

  std::string Dump() {
    std::stringstream res;
    res << error << std::endl;

    return res.str();
  }

  [[noreturn]] void Abort() {
    std::cerr << error << std::endl;
    std::abort();
  }

 private:
  ErrorType error;
};

}  // namespace simple

}  // namespace maybe

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_ERROR_H_
