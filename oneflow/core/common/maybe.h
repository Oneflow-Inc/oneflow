#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/either_ptr.h"
#include "oneflow/core/common/error.h"

namespace oneflow {

class ErrorMessage {
 public:
  ErrorMessage() : error_(std::make_shared<ErrorProto>()) {}
  virtual ~ErrorMessage() = default;

  template<typename MessageType>
  ErrorMessage&& operator<<(const MessageType& msg) {
    oss_ << msg;
    return std::move(*this);
  }

  std::shared_ptr<ErrorProto> Release() const {
    SendMessage();
    return std::move(error_);
  }

  void Reset() {
    ResetStream();
    ResetError();
  }

  OF_DISALLOW_COPY_AND_MOVE(ErrorMessage);

 private:
  void SendMessage() const {
    // TODO: set error type
    error_->set_msg(oss_.str());
  }

  void ResetStream() { oss_.str(""); }

  void ResetError() { error_->Clear(); }

  std::ostringstream oss_;
  mutable std::shared_ptr<ErrorProto> error_;
};

template<typename T>
class MaybeBase {
 public:
  MaybeBase(const std::shared_ptr<T>& data) : data_or_error_(data) {}
  MaybeBase(const std::shared_ptr<ErrorProto>& error) : data_or_error_(error) {}
  MaybeBase(const MaybeBase<T>&) = default;
  ~MaybeBase() = default;  // no virtual is what we want

  bool IsOk() const { return data_or_error_.template Has<T>(); }
  const std::shared_ptr<T>& data() const { return data_or_error_.template Get<T>(); }
  std::shared_ptr<ErrorProto> error() const { return data_or_error_.template Get<ErrorProto>(); }

 private:
  EitherPtr<T, ErrorProto> data_or_error_;
};

template<typename T>
class Maybe final : public MaybeBase<T> {
 public:
  Maybe(const T& data) : MaybeBase<T>(std::make_shared<T>(data)) {}
  Maybe(const Error& error) : MaybeBase<T>(error.error_proto()) {}
  Maybe(const std::shared_ptr<T>& data) : MaybeBase<T>(data) {}
  Maybe(const std::shared_ptr<ErrorProto>& error) : MaybeBase<T>(error) {}
  Maybe(const Maybe<T>&) = default;
  ~Maybe() = default;

  Maybe(const ErrorMessage&& error_msg) : MaybeBase<T>(error_msg.Release()) {}
  static Maybe<T> Ok() { return Maybe<T>(); }
};

template<>
class Maybe<void> final : public MaybeBase<void> {
 public:
  Maybe(const Error& error) : MaybeBase<void>(error.error_proto()) { CheckError(); }
  Maybe(const std::shared_ptr<ErrorProto>& error) : MaybeBase<void>(error) { CheckError(); }
  Maybe(const Maybe<void>&) = default;
  ~Maybe() = default;

  Maybe(const ErrorMessage&& error_msg) : MaybeBase<void>(error_msg.Release()) {
    CheckError();
  }

  static Maybe<void> Ok() { return Maybe<void>(); }

 private:
  Maybe() : MaybeBase<void>(std::shared_ptr<void>()) {}
  void CheckError() const { CHECK_NE(error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET); }
};

template<typename T>
inline Maybe<T> MaybeFuncSafeCallWrapper(Maybe<T>&& maybe) {
  return maybe;
}

#define __MAYBE_CALL_LOC__ __FILE__ ":" OF_PP_STRINGIZE(__LINE__) "\n"

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

#define TRY(...) MaybeFuncSafeCallWrapper(__VA_ARGS__)
#define JUST(...)                                              \
  ({                                                           \
    const auto& maybe = MaybeFuncSafeCallWrapper(__VA_ARGS__); \
    if (!maybe.IsOk()) {                                       \
      LOG(INFO) << "maybe failed:" << __MAYBE_CALL_LOC__;      \
      return maybe.error();                                    \
    }                                                          \
    maybe.data();                                              \
  })
#define CHECK_JUST(...)                                        \
  ({                                                           \
    const auto& maybe = MaybeFuncSafeCallWrapper(__VA_ARGS__); \
    CHECK(maybe.IsOk());                                       \
    maybe.data();                                              \
  })

#else
#error statement expression is no supported, please implement try-catch version of JUST
#endif

}  // namespace oneflow

#define OF_TEST_EQ(lhs, rhs) ((lhs) == (rhs))
#define OF_TEST_GE(lhs, rhs) ((lhs) >= (rhs))
#define OF_TEST_GT(lhs, rhs) ((lhs) > (rhs))
#define OF_TEST_LE(lhs, rhs) ((lhs) <= (rhs))
#define OF_TEST_LT(lhs, rhs) ((lhs) < (rhs))
#define OF_TEST_NE(lhs, rhs) ((lhs) != (rhs))

#define COMPACT_MESSAGE_TAG(time, file, line) COMPACT_MESSAGE_TAG_IMPL(time, file, line)
#define COMPACT_MESSAGE_TAG_IMPL(time, file, line) time " " file ":" #line "] "

#define OF_MESSAGE_TAG COMPACT_MESSAGE_TAG(__TIME__, __FILE__, __LINE__)

#define CHECK_OR_RETURN(expr) \
  if (!(expr)) return ErrorMessage() << OF_MESSAGE_TAG << #expr << ": "

#define ENFORCE_THEN_RETURN(error_type) \
  return ErrorMessage() << OF_MESSAGE_TAG << #error_type << ": "

#define CHECK_EQ_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_EQ(lhs, rhs))

#define CHECK_GE_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_GE(lhs, rhs))

#define CHECK_GT_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_GT(lhs, rhs))

#define CHECK_LE_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_LE(lhs, rhs))

#define CHECK_LT_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_LT(lhs, rhs))

#define CHECK_NE_OR_RETURN(lhs, rhs) CHECK_OR_RETURN(OF_TEST_NE(lhs, rhs))

#define CHECK_STREQ_OR_RETURN(lhs, rhs) CHECK_EQ_OR_RETURN(std::string(lhs), std::string(rhs))

#define UNSUPPORTED_THEN_RETURN() ENFORCE_THEN_RETURN(UNSUPPORTED)

#define TODO_THEN_RETURN() ENFORCE_THEN_RETURN(TODO)

#define UNIMPLEMENTED_THEN_RETURN() ENFORCE_THEN_RETURN(UNIMPLEMENTED)

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
