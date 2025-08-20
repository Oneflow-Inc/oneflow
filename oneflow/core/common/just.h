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

#ifndef ONEFLOW_CORE_COMMON_JUST_H_
#define ONEFLOW_CORE_COMMON_JUST_H_

#include <fmt/format.h>
#include <sstream>
#include <type_traits>
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/maybe/just.h"

namespace oneflow {

template<typename T, typename Enabled = void>
class Maybe;

template<typename T>
class Optional;

Maybe<std::string> FormatErrorStr(const std::shared_ptr<StackedError>&);
namespace {
std::string GetFormatedSerializedError(const std::shared_ptr<StackedError>&);
}

namespace private_details {

template<typename T>
Error&& AddFrameMessage(Error&& error, const T& x) {
  std::ostringstream ss;
  ss << x;
  error->set_frame_msg(error->frame_msg() + ss.str());
  return std::move(error);
}

template<>
inline Error&& AddFrameMessage(Error&& error, const std::stringstream& x) {
  AddFrameMessage(std::move(error), x.str());
  return std::move(error);
}

template<>
inline Error&& AddFrameMessage(Error&& error, const std::ostream& x) {
  AddFrameMessage(std::move(error), x.rdbuf());
  return std::move(error);
}

}  // namespace private_details
}  // namespace oneflow

namespace oneflow::maybe {

template<typename T>
struct JustTraits<::oneflow::Maybe<T>> {
  template<typename U>
  static decltype(auto) Value(U&& v) {
    return std::forward<U>(v).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  };

  template<typename U>
  static decltype(auto) ValueNotFoundError(U&& v) {
    return std::forward<U>(v).stacked_error();
  };
};

template<typename T>
struct JustTraits<::oneflow::Optional<T>> {
  template<typename U>
  static decltype(auto) Value(U&& v) {
    return std::forward<U>(v).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  };

  template<typename U>
  static decltype(auto) ValueNotFoundError(U&&) {
    return Error::ValueNotFoundError().stacked_error();
  };
};

template<>
struct StackedErrorTraits<std::shared_ptr<StackedError>> {
  template<typename T, typename... U, std::enable_if_t<sizeof...(U) == 4, int> = 0>
  static void PushStack(T&& v, U&&... args) {
    auto frame = ::oneflow::SymbolOf(::oneflow::ErrorStackFrame(std::forward<U>(args)...));
    std::forward<T>(v)->add_stack_frame(frame);
  }

  template<typename T, typename U1, typename U2, typename U3, typename U4, typename... U,
           std::enable_if_t<(sizeof...(U) > 0), int> = 0>
  static void PushStack(T&& v, U1&& file, U2&& line, U3&& func, U4&& code_text, U&&... args) {
    PushStack(std::forward<T>(v), std::forward<U1>(file), std::forward<U2>(line),
              std::forward<U3>(func), std::forward<U4>(code_text));

    std::forward<T>(v)->mut_error_proto()->set_frame_msg(fmt::format(
        "{}\nError message from {}:{}\n\t{}: ", std::forward<T>(v)->error_proto()->frame_msg(),
        file, line, code_text));
    (::oneflow::private_details::AddFrameMessage(std::forward<T>(v), std::forward<U>(args)), ...);
    std::forward<T>(v)->mut_error_proto()->set_frame_msg(
        std::forward<T>(v)->error_proto()->frame_msg() + "\n");
  }

  template<typename T>
  [[noreturn]] static void Abort(T&& v) {
    ::oneflow::details::Throw() =
        ::oneflow::Error::RuntimeError().AddStackFrame(std::forward<T>(v)->stack_frame().back())
        << ::oneflow::GetErrorString(std::forward<T>(v));
  }
};

}  // namespace oneflow::maybe

#define TRY(...) JUST_STACK_CHECK_I(__VA_ARGS__)

#endif  // ONEFLOW_CORE_COMMON_JUST_H_
