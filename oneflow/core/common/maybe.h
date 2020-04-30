#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/either_ptr.h"
#include "oneflow/core/common/shared_or_plain.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T>
class Maybe final {
 public:
  Maybe(const T& data) : data_or_error_(std::make_shared<T>(data)) {}
  Maybe(const Error& error) : data_or_error_(error.error_proto()) {}
  Maybe(const std::shared_ptr<T>& data) : data_or_error_(data) {}
  Maybe(const std::shared_ptr<ErrorProto>& error) : data_or_error_(error) {}
  Maybe(const Maybe<T>&) = default;
  Maybe(Maybe<T>&&) = default;
  ~Maybe() = default;

  bool IsOk() const { return data_or_error_.template Has<T>(); }
  std::shared_ptr<T> Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {
    return data_or_error_.template Get<T>();
  }
  std::shared_ptr<ErrorProto> error() const { return data_or_error_.template Get<ErrorProto>(); }

  T GetDataAndSerializedErrorProto(std::string* error_str, const T& default_for_error) const {
    if (IsOk()) {
      google::protobuf::TextFormat::PrintToString(ErrorProto(), error_str);
      return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      google::protobuf::TextFormat::PrintToString(*error(), error_str);
      return default_for_error;
    }
  }

 private:
  EitherPtr<T, ErrorProto> data_or_error_;
};

template<>
class Maybe<void> final {
 public:
  Maybe(const Error& error) : error_or_plain_(error.error_proto()) { CheckError(); }
  Maybe(const std::shared_ptr<ErrorProto>& error) : error_or_plain_(error) { CheckError(); }
  Maybe(const Maybe<void>&) = default;
  Maybe(Maybe<void>&&) = default;
  ~Maybe() = default;

  static Maybe<void> Ok() { return Maybe<void>(); }

  bool IsOk() const { return error_or_plain_.IsPlain(); }
  void Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {}
  std::shared_ptr<ErrorProto> error() const { return error_or_plain_.shared_ptr(); }

  void GetDataAndSerializedErrorProto(std::string* error_str) const {
    if (IsOk()) {
      google::protobuf::TextFormat::PrintToString(ErrorProto(), error_str);
    } else {
      google::protobuf::TextFormat::PrintToString(*error(), error_str);
    }
  }

 private:
  Maybe() : error_or_plain_(nullptr) {}
  void CheckError() const { CHECK_NE(error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET); }

  SharedOrPlain<ErrorProto, void*> error_or_plain_;
};

#define SPECIALIZE_PLAIN_MAYBE(T)                                                                \
  class Maybe<T> final {                                                                         \
   public:                                                                                       \
    Maybe(T data) : error_or_plain_(data) {}                                                     \
    Maybe(const Error& error) : error_or_plain_(error.error_proto()) { CheckError(); }           \
    Maybe(const std::shared_ptr<ErrorProto>& error) : error_or_plain_(error) { CheckError(); }   \
    Maybe(const Maybe<T>&) = default;                                                            \
    Maybe(Maybe<T>&&) = default;                                                                 \
    ~Maybe() = default;                                                                          \
                                                                                                 \
    bool IsOk() const { return error_or_plain_.IsPlain(); }                                      \
    T Data_YouAreNotAllowedToCallThisFuncOutsideThisFile() const {                               \
      return error_or_plain_.plain_data();                                                       \
    }                                                                                            \
    std::shared_ptr<ErrorProto> error() const { return error_or_plain_.shared_ptr(); }           \
                                                                                                 \
    T GetDataAndSerializedErrorProto(std::string* error_str, const T& default_for_error) const { \
      if (IsOk()) {                                                                              \
        google::protobuf::TextFormat::PrintToString(ErrorProto(), error_str);                    \
        return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();                             \
      } else {                                                                                   \
        google::protobuf::TextFormat::PrintToString(*error(), error_str);                        \
        return default_for_error;                                                                \
      }                                                                                          \
    }                                                                                            \
                                                                                                 \
   private:                                                                                      \
    void CheckError() const {                                                                    \
      CHECK_NE(error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);                      \
    }                                                                                            \
                                                                                                 \
    SharedOrPlain<ErrorProto, T> error_or_plain_;                                                \
  }

template<typename T>
SPECIALIZE_PLAIN_MAYBE(T*);

#define SPECIALIZE_BASIC_DATA_TYPE_MAYBE(T) \
  template<>                                \
  SPECIALIZE_PLAIN_MAYBE(T)

SPECIALIZE_BASIC_DATA_TYPE_MAYBE(bool);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(char);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(unsigned char);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(short);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(unsigned short);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(int);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(unsigned int);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(long);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(unsigned long);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(long long);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(unsigned long long);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(float);
SPECIALIZE_BASIC_DATA_TYPE_MAYBE(double);

#undef SPECIALIZE_BASIC_DATA_TYPE_MAYBE
#undef SPECIALIZE_PLAIN_MAYBE

template<typename T>
inline Maybe<T> MaybeFuncSafeCallWrapper(Maybe<T>&& maybe) {
  return maybe;
}

#define __LOC__ __FILE__ ":" OF_PP_STRINGIZE(__LINE__) "\n"

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

#define TRY(...) MaybeFuncSafeCallWrapper(std::move(__VA_ARGS__))
#define JUST(...)                                                         \
  ({                                                                      \
    const auto& maybe = MaybeFuncSafeCallWrapper(std::move(__VA_ARGS__)); \
    if (!maybe.IsOk()) {                                                  \
      LOG(INFO) << "maybe failed:" << __LOC__;                            \
      return maybe.error();                                               \
    }                                                                     \
    maybe.Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();           \
  })
#define CHECK_JUST(...)                                                   \
  ({                                                                      \
    const auto& maybe = MaybeFuncSafeCallWrapper(std::move(__VA_ARGS__)); \
    CHECK(maybe.IsOk());                                                  \
    maybe.Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();           \
  })
#define OF_RETURN_IF_ERROR(...)                                         \
  const auto& maybe = MaybeFuncSafeCallWrapper(std::move(__VA_ARGS__)); \
  if (!maybe.IsOk()) {                                                  \
    LOG(INFO) << "maybe failed:" << __LOC__;                            \
    return maybe.error();                                               \
  }

#else
#error statement expression is no supported, please implement try-catch version of JUST
#endif

}  // namespace oneflow

#define OF_TODO() return __LOC__ <= Error::Todo()
#define OF_UNIMPLEMENTED() return __LOC__ <= Error::Unimplemented()

#define CHECK_OR_RETURN(expr) \
  if (!(expr))                \
  return __LOC__ <= Error::CheckFailed() << " Check failed: " << OF_PP_STRINGIZE(expr) << "\t"

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
