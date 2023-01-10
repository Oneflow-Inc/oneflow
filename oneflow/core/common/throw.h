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
#ifndef ONEFLOW_CORE_COMMON_THROW_H_
#define ONEFLOW_CORE_COMMON_THROW_H_

#include "oneflow/core/common/error.h"

namespace oneflow {

namespace details {

struct Throw final {
  void operator=(Error&& error) { ThrowError(error.stacked_error()); }
};

}  // namespace details

}  // namespace oneflow

#define THROW(err_type)                                                                    \
  ::oneflow::details::Throw() =                                                            \
      ::oneflow::Error::err_type().AddStackFrame([](const char* function) {                \
        thread_local static auto frame =                                                   \
            ::oneflow::SymbolOf(::oneflow::ErrorStackFrame(__FILE__, __LINE__, function)); \
        return frame;                                                                      \
      }(__FUNCTION__))

#define CHECK_OR_THROW(expr)                                                               \
  if (!(expr))                                                                             \
  ::oneflow::details::Throw() =                                                            \
      ::oneflow::Error::CheckFailedError().AddStackFrame([](const char* function) {        \
        thread_local static auto frame =                                                   \
            ::oneflow::SymbolOf(::oneflow::ErrorStackFrame(__FILE__, __LINE__, function)); \
        return frame;                                                                      \
      }(__FUNCTION__))                                                                     \
      << "Check failed: " << OF_PP_STRINGIZE(expr) << ": "

#define CHECK_EQ_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) == (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GE_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) >= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GT_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) > (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LE_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) <= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LT_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) < (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_NE_OR_THROW(lhs, rhs) \
  CHECK_OR_THROW((lhs) != (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_STREQ_OR_THROW(lhs, rhs) CHECK_EQ_OR_THROW(std::string(lhs), std::string(rhs))

#define CHECK_STRNE_OR_THROW(lhs, rhs) CHECK_NE_OR_THROW(std::string(lhs), std::string(rhs))

#define CHECK_NOTNULL_OR_THROW(ptr) CHECK_OR_THROW(ptr != nullptr)

#define CHECK_ISNULL_OR_THROW(ptr) CHECK_OR_THROW(ptr == nullptr)

#define TODO_THEN_THROW()                                                                  \
  ::oneflow::details::Throw() =                                                            \
      ::oneflow::Error::TodoError().AddStackFrame([](const char* function) {               \
        thread_local static auto frame =                                                   \
            ::oneflow::SymbolOf(::oneflow::ErrorStackFrame(__FILE__, __LINE__, function)); \
        return frame;                                                                      \
      }(__FUNCTION__))

#define UNIMPLEMENTED_THEN_THROW()                                                         \
  ::oneflow::details::Throw() =                                                            \
      ::oneflow::Error::UnimplementedError().AddStackFrame([](const char* function) {      \
        thread_local static auto frame =                                                   \
            ::oneflow::SymbolOf(::oneflow::ErrorStackFrame(__FILE__, __LINE__, function)); \
        return frame;                                                                      \
      }(__FUNCTION__))

#endif  // ONEFLOW_CORE_COMMON_THROW_H_
