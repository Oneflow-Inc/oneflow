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
#include <string>
#include <vector>
#include <iostream>

#include "utility.h"
#include "type_traits.h"

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

  static const ErrorType& Error(const T& se) { return se.Error(); }

  static std::size_t StackSize(const T& se) { return se.StackSize(); }

  static ConstRefExceptVoid<StackEntryType> StackElem(const T& se, std::size_t index) {
    return se.StackElem(index);
  }

  template<typename... Args>
  static void PushStack(const T& se, Args&&... args) {
    se.PushStack(std::forward<Args>(args)...);
  }

  [[noreturn]] static void Abort(T& se) { se.Abort(); }
};

// simple implementation for some customization points
namespace simple {

template<typename Message>
struct ErrorStackEntry {
  StringView filename;
  std::size_t lineno;
  StringView function;
  Message message;

  ErrorStackEntry(StringView filename, std::size_t lineno, StringView function, Message message)
      : filename(filename), lineno(lineno), function(function), message(std::move(message)) {}
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
