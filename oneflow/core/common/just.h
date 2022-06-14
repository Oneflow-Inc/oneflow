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

#include <glog/logging.h>
#include <type_traits>
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T, typename Enabled = void>
class Maybe;

template<typename T>
class Optional;

Maybe<std::string> FormatErrorStr(const std::shared_ptr<ErrorProto>&);
namespace {
std::string GetFormatedSerializedError(const std::shared_ptr<ErrorProto>&);
}

namespace private_details {

inline std::shared_ptr<ErrorProto>&& JustErrorAddStackFrame(std::shared_ptr<ErrorProto>&& err,
                                                            const std::string& file, int64_t line,
                                                            const std::string& func,
                                                            const std::string& message) {
  auto* stack_frame = err->add_stack_frame();
  stack_frame->set_file(file);
  stack_frame->set_line(line);
  stack_frame->set_function(func);
  stack_frame->set_error_msg(message);

  return std::move(err);
}

template<typename... T>
Error&& JustErrorAddMessage(Error&& err, T&&... msg) {
  __attribute__((unused)) int dummy[] = {((void)(std::move(err) << std::forward<T>(msg)), 0)...};
  return std::move(err);
}

template<typename T>
bool JustIsOk(const Maybe<T>& val) {
  return val.IsOk();
}

template<typename T>
bool JustIsOk(const Optional<T>& val) {
  return val.has_value();
}

template<typename T>
std::shared_ptr<ErrorProto> JustGetError(const Maybe<T>& val) {
  return val.error();
}

template<typename T>
std::shared_ptr<ErrorProto> JustGetError(const Optional<T>&) {
  return Error::ValueNotFoundError().error_proto();
}

template<typename T>
typename std::remove_const<typename std::remove_reference<T>::type>::type&& RemoveRValConst(
    T&& v) noexcept {
  static_assert(std::is_rvalue_reference<T&&>::value, "rvalue is expected here");
  return const_cast<typename std::remove_const<typename std::remove_reference<T>::type>::type&&>(v);
}

}  // namespace private_details
}  // namespace oneflow

#define __JustStackCheckWrapper__(...) __VA_ARGS__
#define TRY(...) __JustStackCheckWrapper__(__VA_ARGS__)

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

#define JUST(...)                                                                              \
  ::oneflow::private_details::RemoveRValConst(({                                               \
    auto&& _just_value_to_check_ = __JustStackCheckWrapper__(__VA_ARGS__);                     \
    if (!::oneflow::private_details::JustIsOk(_just_value_to_check_)) {                        \
      return ::oneflow::private_details::JustErrorAddStackFrame(                               \
          ::oneflow::private_details::JustGetError(_just_value_to_check_), __FILE__, __LINE__, \
          __FUNCTION__, OF_PP_STRINGIZE(__VA_ARGS__));                                         \
    }                                                                                          \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);                      \
  })).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define CHECK_JUST(...)                                                                            \
  ([&](const char* _just_closure_func_name_) {                                                     \
    auto&& _just_value_to_check_ = __JustStackCheckWrapper__(__VA_ARGS__);                         \
    if (!::oneflow::private_details::JustIsOk(_just_value_to_check_)) {                            \
      LOG(FATAL) << ::oneflow::GetFormatedSerializedError(                                         \
          ::oneflow::private_details::JustErrorAddStackFrame(                                      \
              ::oneflow::private_details::JustGetError(_just_value_to_check_), __FILE__, __LINE__, \
              _just_closure_func_name_, OF_PP_STRINGIZE(__VA_ARGS__)));                            \
    }                                                                                              \
    return std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);                   \
  })(__FUNCTION__)                                                                                 \
      .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define JUST_MSG(value, ...)                                                                \
  ::oneflow::private_details::RemoveRValConst(({                                            \
    auto&& _just_value_to_check_ = (value);                                                 \
    if (!::oneflow::private_details::JustIsOk(_just_value_to_check_)) {                     \
      return ::oneflow::private_details::JustErrorAddMessage(                               \
          ::oneflow::Error(::oneflow::private_details::JustGetError(_just_value_to_check_)) \
              .AddStackFrame(__FILE__, __LINE__, __FUNCTION__),                             \
          OF_PP_STRINGIZE(value), ": ", __VA_ARGS__);                                       \
    }                                                                                       \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);                   \
  })).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define CHECK_JUST_MSG(value, ...)                                                              \
  ([&](const char* _just_closure_func_name_) {                                                  \
    auto&& _just_value_to_check_ = (value);                                                     \
    if (!::oneflow::private_details::JustIsOk(_just_value_to_check_)) {                         \
      LOG(FATAL) << ::oneflow::GetFormatedSerializedError(                                      \
          ::oneflow::private_details::JustErrorAddMessage(                                      \
              ::oneflow::Error(::oneflow::private_details::JustGetError(_just_value_to_check_)) \
                  .AddStackFrame(__FILE__, __LINE__, _just_closure_func_name_),                 \
              OF_PP_STRINGIZE(value), ": ", __VA_ARGS__)                                        \
              .error_proto());                                                                  \
    }                                                                                           \
    return std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);                \
  })(__FUNCTION__)                                                                              \
      .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define JUST_OPT(...)                                                      \
  ::oneflow::private_details::RemoveRValConst(({                           \
    auto&& _just_value_to_check_ = __JustStackCheckWrapper__(__VA_ARGS__); \
    if (!_just_value_to_check_.has_value()) { return NullOpt; }            \
    std::forward<decltype(_just_value_to_check_)>(_just_value_to_check_);  \
  })).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#else
#error statement expression is no supported, please implement try-catch version of JUST
#endif

#endif  // ONEFLOW_CORE_COMMON_JUST_H_
