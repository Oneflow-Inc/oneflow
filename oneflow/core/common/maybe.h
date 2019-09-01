#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/either_ptr.h"
#include "oneflow/core/common/error.pb.h"

namespace oneflow {

template<typename T>
class MaybeBase {
 public:
  MaybeBase(const std::shared_ptr<const Error>& error) : data_or_error_(error) {}
  MaybeBase(const std::shared_ptr<T>& data) : data_or_error_(data) {}
  MaybeBase(const MaybeBase<T>&) = default;
  virtual ~MaybeBase() = default;

  bool IsOk() const { return data_or_error_.template Has<T>(); }
  const std::shared_ptr<T>& data() const { return data_or_error_.template Get<T>(); }
  std::shared_ptr<const Error> error() const { return data_or_error_.template Get<const Error>(); }

 private:
  EitherPtr<T, const Error> data_or_error_;
};

template<typename T>
class Maybe final : public MaybeBase<T> {
 public:
  Maybe(const Error& error) : MaybeBase<T>(std::make_shared<const Error>(error)) {}
  Maybe(const T& data) : MaybeBase<T>(std::make_shared<T>(data)) {}
  Maybe(const std::shared_ptr<const Error>& error) : MaybeBase<T>(error) {}
  Maybe(const std::shared_ptr<T>& data) : MaybeBase<T>(data) {}
  Maybe(const Error* error) : MaybeBase<T>(std::shared_ptr<const Error>(error)) {}
  Maybe(T* data) : MaybeBase<T>(std::shared_ptr<T>(data)) {}
  Maybe(const Maybe<T>&) = default;
  ~Maybe() override = default;

  static Maybe<T> Ok() { return Maybe<T>(); }
};

template<>
class Maybe<void> final : public MaybeBase<void> {
 public:
  Maybe(const Error& error) : MaybeBase<void>(std::make_shared<const Error>(error)) {
    CheckError();
  }
  Maybe(const std::shared_ptr<const Error>& error) : MaybeBase<void>(error) { CheckError(); }
  Maybe(const Error* error) : MaybeBase<void>(std::shared_ptr<const Error>(error)) { CheckError(); }
  Maybe(const Maybe<void>&) = default;
  ~Maybe() override = default;

  static Maybe<void> Ok() { return Maybe<void>(); }

 private:
  Maybe() : MaybeBase<void>(std::shared_ptr<void>()) {}
  void CheckError() const { CHECK_NE(error()->error_type_case(), Error::ERROR_TYPE_NOT_SET); }
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

namespace {

enum class ErrorType {
  kUnknown = 0,
  kCondition = 1,
  kEnforce = 2,
};

template<ErrorType type>
std::ostringstream& SerializeExprError(std::ostringstream& oss, const std::string& expr) {
  oss << "Unknown type error `" << expr << "` occurs.";
  return oss;
}

template<>
std::ostringstream& SerializeExprError<ErrorType::kCondition>(std::ostringstream& oss,
                                                              const std::string& expr) {
  oss << "Condition expression `" << expr << "` check failed.";
  return oss;
}

template<>
std::ostringstream& SerializeExprError<ErrorType::kEnforce>(std::ostringstream& oss,
                                                            const std::string& expr) {
  oss << "Enforce error `" << expr << "` occurs.";
  return oss;
}

std::string Sprintf() { return ""; }

template<typename... Args>
std::string Sprintf(const Args&... args) {
  char buffer[2048];
  snprintf(buffer, sizeof(buffer), std::forward<const Args>(args)...);
  return std::string(buffer);
}

}  // namespace

#define OF_TEST_EQ(lhs, rhs) ((lhs) == (rhs))
#define OF_TEST_GE(lhs, rhs) ((lhs) >= (rhs))
#define OF_TEST_GT(lhs, rhs) ((lhs) > (rhs))
#define OF_TEST_LE(lhs, rhs) ((lhs) <= (rhs))
#define OF_TEST_LT(lhs, rhs) ((lhs) < (rhs))
#define OF_TEST_NE(lhs, rhs) ((lhs) != (rhs))

#define GEN_ERROR_MSG(type, expr, ...)             \
  [&]() -> std::string {                           \
    std::string detail = Sprintf(__VA_ARGS__);     \
    std::ostringstream oss;                        \
    SerializeExprError<type>(oss, expr);           \
    if (!detail.empty()) { oss << " " << detail; } \
    return oss.str();                              \
  }()

#define CHECK_OR_RETURN(expr, ...)                                             \
  {                                                                            \
    if (!(expr)) {                                                             \
      Error error;                                                             \
      error.set_msg(GEN_ERROR_MSG(ErrorType::kCondition, #expr, __VA_ARGS__)); \
      return Maybe<void>(error);                                               \
    }                                                                          \
  }

#define CHECK_EQ_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_EQ(lhs, rhs), __VA_ARGS__)

#define CHECK_GE_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_GE(lhs, rhs), __VA_ARGS__)

#define CHECK_GT_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_GT(lhs, rhs), __VA_ARGS__)

#define CHECK_LE_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_LE(lhs, rhs), __VA_ARGS__)

#define CHECK_LT_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_LT(lhs, rhs), __VA_ARGS__)

#define CHECK_NE_OR_RETURN(lhs, rhs, ...) CHECK_OR_RETURN(OF_TEST_NE(lhs, rhs), __VA_ARGS__)

#define CHECK_STREQ_OR_RETURN(lhs, rhs, ...) \
  CHECK_EQ_OR_RETURN(std::string(lhs), std::string(rhs), __VA_ARGS__)

#define ENFORCE_THEN_RETURN(type, ...)                                     \
  {                                                                        \
    Error error;                                                           \
    error.set_msg(GEN_ERROR_MSG(ErrorType::kEnforce, #type, __VA_ARGS__)); \
    return Maybe<void>(error);                                             \
  }

#define UNSUPPORTED_THEN_RETURN(...) ENFORCE_THEN_RETURN(OF_TEST_UNSUPPORTED, __VA_ARGS__)

#define TODO_THEN_RETURN(...) ENFORCE_THEN_RETURN(OF_TEST_TODO, __VA_ARGS__)

#define UNIMPLEMENTED_THEN_RETURN(...) ENFORCE_THEN_RETURN(OF_TEST_UNIMPLEMENTED, __VA_ARGS__)

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
