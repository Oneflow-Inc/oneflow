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

namespace oneflow {

template<typename T, typename Enabled = void>
class Maybe;

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
  Maybe(const Error& error) : data_or_error_(error.error_proto()) {}
  Maybe(const std::shared_ptr<T>& data) : data_or_error_(data) {}
  Maybe(std::shared_ptr<T>&& data) : data_or_error_(std::move(data)) {}
  Maybe(const std::shared_ptr<cfg::ErrorProto>& error) : data_or_error_(error) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&& other) : data_or_error_(std::move(other.data_or_error_)) {}
  ~Maybe() = default;

  bool IsOk() const { return data_or_error_.template Has<T>(); }
  std::shared_ptr<T> Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return data_or_error_.template Get<T>();
  }
  std::shared_ptr<cfg::ErrorProto> error() const {
    return data_or_error_.template Get<cfg::ErrorProto>();
  }

  std::string GetSerializedError() const { return this->error()->DebugString(); }

  template<typename Type = T>
  Type GetDataAndSerializedErrorProto(std::string* error_str, const Type& default_for_error) const {
    static_assert(std::is_same<T, Type>::value, "error type for argument 1");
    if (IsOk()) {
      *error_str = cfg::ErrorProto().DebugString();
      return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      *error_str = this->error()->DebugString();
      return default_for_error;
    }
  }

  template<typename Type = T>
  std::pair<Type, std::shared_ptr<cfg::ErrorProto>> GetDataAndErrorProto(
      const Type& default_for_error) const {
    if (IsOk()) {
      return std::make_pair(*Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<cfg::ErrorProto>());
    } else {
      return std::make_pair(default_for_error, error());
    }
  }

  std::pair<std::shared_ptr<T>, std::shared_ptr<cfg::ErrorProto>> GetDataPtrAndErrorProto() const {
    if (IsOk()) {
      return std::make_pair(Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<cfg::ErrorProto>());
    } else {
      return std::make_pair(std::shared_ptr<T>(), error());
    }
  }

  template<typename Type = T>
  Type GetOrThrow() const {
    if (!IsOk()) { ThrowError(error()); }
    return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

  std::shared_ptr<T> GetPtrOrThrow() const {
    if (!IsOk()) { ThrowError(error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  EitherPtr<T, cfg::ErrorProto> data_or_error_;
};

template<typename T>
class Maybe<T, typename std::enable_if<std::is_same<T, void>::value>::type> final {
 public:
  Maybe(const Error& error) : error_or_scalar_(error.error_proto()) { CheckError(); }
  Maybe(const std::shared_ptr<cfg::ErrorProto>& error) : error_or_scalar_(error) { CheckError(); }
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  static Maybe Ok() { return Maybe(); }

  bool IsOk() const { return error_or_scalar_.IsScalar(); }
  void Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {}
  std::shared_ptr<cfg::ErrorProto> error() const { return error_or_scalar_.shared_ptr(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return this->error()->DebugString();
  }

  void GetDataAndSerializedErrorProto(std::string* error_str) const {
    if (IsOk()) {
      *error_str = cfg::ErrorProto().DebugString();
    } else {
      *error_str = this->error()->DebugString();
    }
  }

  std::shared_ptr<cfg::ErrorProto> GetDataAndErrorProto() const {
    if (IsOk()) {
      return std::shared_ptr<cfg::ErrorProto>();
    } else {
      return error();
    }
  }

  void GetOrThrow() const {
    if (!IsOk()) { ThrowError(error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  Maybe() : error_or_scalar_(nullptr) {}
  void CheckError() const {
    CHECK_NE(this->error()->error_type_case(), cfg::ErrorProto::ERROR_TYPE_NOT_SET);
  }

  SharedOrScalar<cfg::ErrorProto, void*> error_or_scalar_;
};

inline const std::shared_ptr<cfg::ErrorProto>& UninitializedValueError() {
  static thread_local const auto& error = Error::ValueError("uninitialized value").error_proto();
  return error;
}

template<typename T>
class Maybe<T, typename std::enable_if<IsScalarType<T>::value>::type> final {
 public:
  Maybe(T data) : error_or_scalar_(data) {}
  Maybe(const Error& error) : error_or_scalar_(error.error_proto()) { CheckError(); }
  Maybe(const std::shared_ptr<cfg::ErrorProto>& error) : error_or_scalar_(error) { CheckError(); }
  Maybe() : error_or_scalar_(UninitializedValueError()) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  void operator=(const Maybe& rhs) { error_or_scalar_ = rhs.error_or_scalar_; }

  bool IsOk() const { return error_or_scalar_.IsScalar(); }
  T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return error_or_scalar_.scalar_value();
  }
  std::shared_ptr<cfg::ErrorProto> error() const { return error_or_scalar_.shared_ptr(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return this->error()->DebugString();
  }

  T GetDataAndSerializedErrorProto(std::string* error_str, const T& default_for_error) const {
    if (IsOk()) {
      *error_str = cfg::ErrorProto().DebugString();
      return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      *error_str = this->error()->DebugString();
      return default_for_error;
    }
  }

  std::pair<T, std::shared_ptr<cfg::ErrorProto>> GetDataAndErrorProto(
      const T& default_for_error) const {
    if (IsOk()) {
      return std::make_pair(Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<cfg::ErrorProto>());
    } else {
      return std::make_pair(default_for_error, error());
    }
  }

  T GetOrThrow() const {
    if (!IsOk()) { ThrowError(error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  void CheckError() const {
    CHECK_NE(this->error()->error_type_case(), cfg::ErrorProto::ERROR_TYPE_NOT_SET);
  }

  SharedOrScalar<cfg::ErrorProto, T> error_or_scalar_;
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
  Maybe(const std::shared_ptr<cfg::ErrorProto>& error) : maybe_ptr_(error) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&&) = default;
  ~Maybe() = default;

  bool IsOk() const { return maybe_ptr_.IsOk(); }
  T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return *maybe_ptr_.Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }
  std::shared_ptr<cfg::ErrorProto> error() const { return maybe_ptr_.error(); }

  std::string GetSerializedError() const { return maybe_ptr_.GetSerializedError(); }

  T GetDataAndSerializedErrorProto(std::string* error_str) const {
    return *maybe_ptr_.GetDataAndSerializedErrorProto(error_str, static_cast<PtrT>(nullptr));
  }

  T GetOrThrow() const {
    if (!IsOk()) { ThrowError(error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  Maybe<PtrT> maybe_ptr_;
};

#define __MaybeErrorStackCheckWrapper__(...) __VA_ARGS__

inline bool MaybeIsOk(Maybe<void>&& maybe) {
  if (!maybe.IsOk()) { LOG(ERROR) << "\n" << maybe.GetSerializedError(); }
  return maybe.IsOk();
}

#define MAYBE_FAILED_LOC __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

namespace private_details {

inline void MaybeErrorAddStackFrame(const std::shared_ptr<cfg::ErrorProto>& err,
                                    const std::string& file, int64_t line, const std::string& func,
                                    const std::string& message) {
  auto* stack_frame = err->add_stack_frame();
  stack_frame->set_file(file);
  stack_frame->set_line(line);
  stack_frame->set_function(func);
  stack_frame->set_error_msg(message);
}

template<typename... T>
Error&& MaybeErrorAddMessage(Error&& err, T&&... msg) {
  __attribute__((unused)) int dummy[] = {((void)(std::move(err) << std::forward<T>(msg)), 0)...};
  return std::move(err);
}

}  // namespace private_details

#define TRY(...) __MaybeErrorStackCheckWrapper__(__VA_ARGS__)
#define JUST(...)                                                                           \
  ({                                                                                        \
    auto&& maybe = __MaybeErrorStackCheckWrapper__(__VA_ARGS__);                            \
    if (!maybe.IsOk()) {                                                                    \
      ::oneflow::private_details::MaybeErrorAddStackFrame(                                  \
          maybe.error(), __FILE__, __LINE__, __FUNCTION__, OF_PP_STRINGIZE((__VA_ARGS__))); \
      return maybe.error();                                                                 \
    }                                                                                       \
    std::move(maybe);                                                                       \
  }).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()
#define CHECK_JUST(...)                                                                  \
  ([&](const char* func_name) {                                                          \
    auto&& maybe = __MaybeErrorStackCheckWrapper__(__VA_ARGS__);                         \
    if (!maybe.IsOk()) {                                                                 \
      ::oneflow::private_details::MaybeErrorAddStackFrame(                               \
          maybe.error(), __FILE__, __LINE__, func_name, OF_PP_STRINGIZE((__VA_ARGS__))); \
      LOG(FATAL) << maybe.GetSerializedError();                                          \
    }                                                                                    \
    return std::move(maybe);                                                             \
  })(__FUNCTION__)                                                                       \
      .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define JUST_MSG(value, ...)                                                               \
  ({                                                                                       \
    auto&& maybe = (value);                                                                \
    if (!maybe.IsOk()) {                                                                   \
      return ::oneflow::private_details::MaybeErrorAddMessage(                             \
          ::oneflow::Error(maybe.error()).AddStackFrame(__FILE__, __LINE__, __FUNCTION__), \
          OF_PP_STRINGIZE((value)), ": ", __VA_ARGS__);                                    \
    }                                                                                      \
    std::move(maybe);                                                                      \
  }).Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define CHECK_JUST_MSG(value, ...)                                                             \
  ([&](const char* func_name) {                                                                \
    auto&& maybe = (value);                                                                    \
    if (!maybe.IsOk()) {                                                                       \
      LOG(FATAL)                                                                               \
          << ::oneflow::private_details::MaybeErrorAddMessage(                                 \
                 ::oneflow::Error(maybe.error()).AddStackFrame(__FILE__, __LINE__, func_name), \
                 OF_PP_STRINGIZE((value)), ": ", __VA_ARGS__)                                  \
                 ->DebugString();                                                              \
    }                                                                                          \
    return std::move(maybe);                                                                   \
  })(__FUNCTION__)                                                                             \
      .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()

#define CHECK_OK(...) CHECK(MaybeIsOk(__VA_ARGS__))

#define OF_RETURN_IF_ERROR(...)                                                \
  for (auto&& maybe_##__LINE__ = __MaybeErrorStackCheckWrapper__(__VA_ARGS__); \
       !maybe_##__LINE__.IsOk();)                                              \
  return Error(maybe_##__LINE__.error()).AddStackFrame(__FILE__, __LINE__, __FUNCTION__)

#else
#error statement expression is no supported, please implement try-catch version of JUST
#endif

}  // namespace oneflow

#define OF_TODO() return Error::Todo().AddStackFrame(__FILE__, __LINE__, __FUNCTION__)
#define OF_UNIMPLEMENTED() \
  return Error::Unimplemented().AddStackFrame(__FILE__, __LINE__, __FUNCTION__)

#define OF_RUNTIME_ERROR()                                                                        \
  return Error::RuntimeError().AddStackFrame(__FILE__, __LINE__, __FUNCTION__) << "RuntimeError " \
                                                                                  ": "

#define OF_COMPLIE_OPTION_ERROR()                                                    \
  return Error::CompileOptionWrong().AddStackFrame(__FILE__, __LINE__, __FUNCTION__) \
         << " Compile option wrong: "

#define CHECK_OR_RETURN(expr)                                                      \
  if (!(expr))                                                                     \
  return Error::CheckFailedError().AddStackFrame(__FILE__, __LINE__, __FUNCTION__) \
         << " Check failed: " << OF_PP_STRINGIZE(expr) << " "

#define CHECK_EQ_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) == (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) >= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_GT_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) > (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) <= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_LT_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) < (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_NE_OR_RETURN(lhs, rhs) \
  CHECK_OR_RETURN((lhs) != (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define CHECK_STREQ_OR_RETURN(lhs, rhs) CHECK_EQ_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_STRNE_OR_RETURN(lhs, rhs) CHECK_NE_OR_RETURN(std::string(lhs), std::string(rhs))

#define CHECK_NOTNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr != nullptr)

#define CHECK_ISNULL_OR_RETURN(ptr) CHECK_OR_RETURN(ptr == nullptr)

#define TODO_THEN_RETURN() OF_TODO()

#define UNIMPLEMENTED_THEN_RETURN() OF_UNIMPLEMENTED()

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
