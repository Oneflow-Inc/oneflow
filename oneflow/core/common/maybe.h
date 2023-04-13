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
#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/either_ptr.h"
#include "oneflow/core/common/shared_or_scalar.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/just.h"

namespace oneflow {

template<typename T>
struct is_maybe {
  static const bool value = false;
};

template<typename T>
struct is_maybe<Maybe<T>> {
  static const bool value = true;
};

template<typename T>
class Maybe<T, typename std::enable_if<!(std::is_same<T, void>::value || IsScalarType<T>::value)
                                       && !std::is_reference<T>::value>::type>
    final {
 public:
  Maybe(const T& data) : data_or_error_(std::make_shared<T>(data)) {}
  Maybe(T&& data) : data_or_error_(std::make_shared<T>(std::move(data))) {}
  Maybe(const Error& error) : data_or_error_(error.stacked_error()) {}
  Maybe(const std::shared_ptr<T>& data) : data_or_error_(data) {}
  Maybe(std::shared_ptr<T>&& data) : data_or_error_(std::move(data)) {}
  Maybe(const std::shared_ptr<StackedError>& error) : data_or_error_(error) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&& other) : data_or_error_(std::move(other.data_or_error_)) {}
  ~Maybe() = default;

  bool IsOk() const { return data_or_error_.template Has<T>(); }
  std::shared_ptr<T> Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return data_or_error_.template Get<T>();
  }
  std::shared_ptr<StackedError> stacked_error() const {
    return data_or_error_.template Get<StackedError>();
  }
  std::shared_ptr<const ErrorProto> error() const { return stacked_error()->error_proto(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return GetFormatedSerializedError(this->stacked_error());
  }

  template<typename Type = T>
  Type GetDataAndSerializedStackedError(std::string* error_str,
                                        const Type& default_for_error) const {
    static_assert(std::is_same<T, Type>::value, "error type for argument 1");
    if (IsOk()) {
      *error_str = StackedError().DebugString();
      return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      *error_str = this->stacked_error()->DebugString();
      return default_for_error;
    }
  }

  template<typename Type = T>
  std::pair<Type, std::shared_ptr<StackedError>> GetDataAndStackedError(
      const Type& default_for_error) const {
    if (IsOk()) {
      return std::make_pair(*Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<StackedError>());
    } else {
      return std::make_pair(default_for_error, stacked_error());
    }
  }

  std::pair<std::shared_ptr<T>, std::shared_ptr<StackedError>> GetDataPtrAndStackedError() const {
    if (IsOk()) {
      return std::make_pair(Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<StackedError>());
    } else {
      return std::make_pair(std::shared_ptr<T>(), stacked_error());
    }
  }

  template<typename Type = T>
  Type GetOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

  std::shared_ptr<T> GetPtrOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  EitherPtr<T, StackedError> data_or_error_;
};

template<typename T>
class Maybe<T, typename std::enable_if<std::is_same<T, void>::value>::type> final {
 public:
  Maybe(const Error& error) : error_or_scalar_(error.stacked_error()) { CheckError(); }
  Maybe(const std::shared_ptr<StackedError>& error) : error_or_scalar_(error) { CheckError(); }
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  static Maybe Ok() { return Maybe(); }

  bool IsOk() const { return error_or_scalar_.IsScalar(); }
  void Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {}
  std::shared_ptr<StackedError> stacked_error() const { return error_or_scalar_.shared_ptr(); }
  std::shared_ptr<const ErrorProto> error() const { return stacked_error()->error_proto(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return GetFormatedSerializedError(this->stacked_error());
  }

  void GetDataAndSerializedStackedError(std::string* error_str) const {
    if (IsOk()) {
      *error_str = StackedError().DebugString();
    } else {
      *error_str = this->stacked_error()->DebugString();
    }
  }

  std::shared_ptr<StackedError> GetDataAndStackedError() const {
    if (IsOk()) {
      return std::shared_ptr<StackedError>();
    } else {
      return stacked_error();
    }
  }

  void GetOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  Maybe() : error_or_scalar_(nullptr) {}
  void CheckError() const {
    CHECK_NE(this->error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);
  }

  SharedOrScalar<StackedError, void*> error_or_scalar_;
};

inline const std::shared_ptr<StackedError>& UninitializedValueError() {
  static thread_local const auto& error =
      (Error::InvalidValueError() << "uninitialized value").stacked_error();
  return error;
}

template<typename T>
class Maybe<T, typename std::enable_if<IsScalarType<T>::value>::type> final {
 public:
  Maybe(T data) : error_or_scalar_(data) {}
  Maybe(const Error& error) : error_or_scalar_(error.stacked_error()) { CheckError(); }
  Maybe(const std::shared_ptr<StackedError>& error) : error_or_scalar_(error) { CheckError(); }
  Maybe() : error_or_scalar_(UninitializedValueError()) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  void operator=(const Maybe& rhs) { error_or_scalar_ = rhs.error_or_scalar_; }

  bool IsOk() const { return error_or_scalar_.IsScalar(); }
  T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return error_or_scalar_.scalar_value();
  }
  std::shared_ptr<StackedError> stacked_error() const { return error_or_scalar_.shared_ptr(); }
  std::shared_ptr<const ErrorProto> error() const { return stacked_error()->error_proto(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return GetFormatedSerializedError(this->stacked_error());
  }

  T GetDataAndSerializedStackedError(std::string* error_str, const T& default_for_error) const {
    if (IsOk()) {
      *error_str = StackedError().DebugString();
      return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      *error_str = this->stacked_error()->DebugString();
      return default_for_error;
    }
  }

  std::pair<T, std::shared_ptr<StackedError>> GetDataAndStackedError(
      const T& default_for_error) const {
    if (IsOk()) {
      return std::make_pair(Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<StackedError>());
    } else {
      return std::make_pair(default_for_error, stacked_error());
    }
  }

  T GetOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  void CheckError() const {
    CHECK_NE(this->error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);
  }

  SharedOrScalar<StackedError, T> error_or_scalar_;
};

template<typename T>
class Maybe<T, typename std::enable_if<!(std::is_same<T, void>::value || IsScalarType<T>::value)
                                       && std::is_reference<T>::value>::type>
    final {
  using ValueT = typename std::remove_reference<T>::type;
  using PtrT = ValueT*;

 public:
  Maybe(T data) : maybe_ptr_(&data) {}
  Maybe(const Error& error) : maybe_ptr_(error) {}
  Maybe(const std::shared_ptr<StackedError>& error) : maybe_ptr_(error) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  bool IsOk() const { return maybe_ptr_.IsOk(); }
  T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return *maybe_ptr_.Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }
  std::shared_ptr<StackedError> stacked_error() const { return maybe_ptr_.stacked_error(); }
  std::shared_ptr<const ErrorProto> error() const { return stacked_error()->error_proto(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return maybe_ptr_.GetSerializedError();
  }

  T GetDataAndSerializedStackedError(std::string* error_str) const {
    return *maybe_ptr_.GetDataAndSerializedStackedError(error_str, static_cast<PtrT>(nullptr));
  }

  T GetOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  Maybe<PtrT> maybe_ptr_;
};

namespace {
std::string GetFormatedSerializedError(const std::shared_ptr<StackedError>& stacked_error) {
  // return error msg got from formatted function or debugstring.
  const auto& maybe_error = TRY(FormatErrorStr(stacked_error));
  const auto& error_str = maybe_error.GetDataAndStackedError(stacked_error->DebugString());
  return error_str.first;
}
}  // namespace
}  // namespace oneflow

#define CHECK_OK(...)                                         \
  for (auto&& maybe = __JustStackCheckWrapper__(__VA_ARGS__); \
       GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!maybe.IsOk());)       \
  LOG(FATAL) << OF_PP_STRINGIZE(__VA_ARGS__) << " is not OK:\n" << maybe.GetSerializedError()

#define OF_RETURN_IF_ERROR(...)                                                               \
  for (auto&& maybe_##__LINE__ = __JustStackCheckWrapper__(__VA_ARGS__);                      \
       !maybe_##__LINE__.IsOk();)                                                             \
  return Error(maybe_##__LINE__.stacked_error()).AddStackFrame([](const char* function) {     \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))

#define OF_TODO()                                                                             \
  return Error::TodoError().AddStackFrame([](const char* function) {                          \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))
#define OF_UNIMPLEMENTED()                                                                    \
  return Error::UnimplementedError().AddStackFrame([](const char* function) {                 \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))

#define OF_RUNTIME_ERROR()                                                                    \
  return Error::RuntimeError().AddStackFrame([](const char* function) {                       \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))                                                                            \
         << "RuntimeError "                                                                   \
            ": "
#define RETURN_ERROR_WITH_BUG_PROMPT() OF_RUNTIME_ERROR() << kOfBugIssueUploadPrompt

#define OF_LOG_ONCE(x)          \
  {                             \
    static bool warned = false; \
    if (!warned) {              \
      warned = true;            \
      x;                        \
    }                           \
  }

#define OF_COMPLIE_OPTION_ERROR()                                                             \
  return Error::CompileOptionWrongError().AddStackFrame([](const char* function) {            \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))                                                                            \
         << "Compile option wrong: "

#define CHECK_OR_RETURN_INTERNAL(expr, error_msg)                           \
  if (!(expr))                                                              \
  return Error::CheckFailedError().AddStackFrame([](const char* function) { \
    thread_local static auto frame =                                        \
        SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function, error_msg)); \
    return frame;                                                           \
  }(__FUNCTION__))

#define CHECK_OR_RETURN_ERROR(expr)                                                           \
  if (!(expr))                                                                                \
  return Error::CheckFailedError().AddStackFrame([](const char* function) {                   \
    thread_local static auto frame = SymbolOf(ErrorStackFrame(__FILE__, __LINE__, function)); \
    return frame;                                                                             \
  }(__FUNCTION__))

// NOTE: Please contact @daquexian if you need to modify these CHECK_(XX_)OR_RETURN macros. There
// are some static analyzers depending on the internal implementation of them.
#define CHECK_OR_RETURN(expr)                                            \
  CHECK_OR_RETURN_INTERNAL(expr, OF_PP_STRINGIZE(CHECK_OR_RETURN(expr))) \
      << "Check failed: (" << OF_PP_STRINGIZE(expr) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_EQ_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) == (rhs), OF_PP_STRINGIZE(CHECK_EQ_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " == " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_GE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) >= (rhs), OF_PP_STRINGIZE(CHECK_GE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " >= " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_GT_OR_RETURN(lhs, rhs)                                                     \
  CHECK_OR_RETURN_INTERNAL((lhs) > (rhs), OF_PP_STRINGIZE(CHECK_GT_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " > " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_LE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) <= (rhs), OF_PP_STRINGIZE(CHECK_LE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " <= " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_LT_OR_RETURN(lhs, rhs)                                                     \
  CHECK_OR_RETURN_INTERNAL((lhs) < (rhs), OF_PP_STRINGIZE(CHECK_LT_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " < " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_NE_OR_RETURN(lhs, rhs)                                                      \
  CHECK_OR_RETURN_INTERNAL((lhs) != (rhs), OF_PP_STRINGIZE(CHECK_NE_OR_RETURN(lhs, rhs))) \
      << "Check failed: (" << (lhs) << " != " << (rhs) << ") " << Error::kOverrideThenMergeMessage

#define CHECK_STREQ_OR_RETURN(lhs, rhs) CHECK_EQ_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_STRNE_OR_RETURN(lhs, rhs) CHECK_NE_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_NOTNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr != nullptr)

#define CHECK_ISNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr == nullptr)

#define TODO_THEN_RETURN() OF_TODO()

#define UNIMPLEMENTED_THEN_RETURN() OF_UNIMPLEMENTED()

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
