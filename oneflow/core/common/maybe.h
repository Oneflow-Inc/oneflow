#ifndef ONEFLOW_CORE_COMMON_MAYBE_H_
#define ONEFLOW_CORE_COMMON_MAYBE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/either_ptr.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

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

  static Maybe<T> Ok() { return Maybe<T>(); }
};

template<>
class Maybe<void> final : public MaybeBase<void> {
 public:
  Maybe(const Error& error) : MaybeBase<void>(error.error_proto()) { CheckError(); }
  Maybe(const std::shared_ptr<ErrorProto>& error) : MaybeBase<void>(error) { CheckError(); }
  Maybe(const Maybe<void>&) = default;
  ~Maybe() = default;

  static Maybe<void> Ok() { return Maybe<void>(); }

 private:
  Maybe() : MaybeBase<void>(std::shared_ptr<void>()) {}
  void CheckError() const { CHECK_NE(error()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET); }
};

template<typename T>
inline Maybe<T> MaybeFuncSafeCallWrapper(Maybe<T>&& maybe) {
  return maybe;
}

#define __LOC__ __FILE__ ":" OF_PP_STRINGIZE(__LINE__) "\n"

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__clang__)

#define TRY(...) MaybeFuncSafeCallWrapper(__VA_ARGS__)
#define JUST(...)                                              \
  ({                                                           \
    const auto& maybe = MaybeFuncSafeCallWrapper(__VA_ARGS__); \
    if (!maybe.IsOk()) {                                       \
      LOG(INFO) << "maybe failed:" << __LOC__;                 \
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

#define OF_CHECK(expr) \
  if (!(expr))         \
  return __LOC__ <= Error::CheckFailed() << " Check failed: " << OF_PP_STRINGIZE(expr) << "\t"

#define OF_CHECK_NOTNULL(ptr) OF_CHECK(ptr != nullptr)
#define OF_CHECK_ISNULL(ptr) OF_CHECK(ptr == nullptr)
#define OF_CHECK_STREQ(lhs, rhs) OF_CHECK_EQ(std::string(lhs), std::string(rhs))
#define OF_CHECK_STRNE(lhs, rhs) OF_CHECK_NE(std::string(lhs), std::string(rhs))

#define OF_CHECK_EQ(lhs, rhs) OF_CHECK((lhs) == (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "
#define OF_CHECK_NE(lhs, rhs) OF_CHECK((lhs) != (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "
#define OF_CHECK_GT(lhs, rhs) OF_CHECK((lhs) > (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "
#define OF_CHECK_GE(lhs, rhs) OF_CHECK((lhs) >= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "
#define OF_CHECK_LT(lhs, rhs) OF_CHECK((lhs) < (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "
#define OF_CHECK_LE(lhs, rhs) OF_CHECK((lhs) <= (rhs)) << "(" << (lhs) << " vs " << (rhs) << ") "

#define OF_TODO() return __LOC__ <= Error::Todo()
#define OF_UNIMPLEMENTED() return __LOC__ <= Error::Unimplemented()

#define CHECK_OR_RETURN(expr) OF_CHECK(expr)

#define CHECK_EQ_OR_RETURN(lhs, rhs) OF_CHECK_EQ(lhs, rhs)

#define CHECK_GE_OR_RETURN(lhs, rhs) OF_CHECK_GE(lhs, rhs)

#define CHECK_GT_OR_RETURN(lhs, rhs) OF_CHECK_GT(lhs, rhs)

#define CHECK_LE_OR_RETURN(lhs, rhs) OF_CHECK_LE(lhs, rhs)

#define CHECK_LT_OR_RETURN(lhs, rhs) OF_CHECK_LT(lhs, rhs)

#define CHECK_NE_OR_RETURN(lhs, rhs) OF_CHECK_NE(lhs, rhs)

#define CHECK_STREQ_OR_RETURN(lhs, rhs) OF_CHECK_STREQ(lhs, rhs)

#define TODO_THEN_RETURN() OF_TODO()

#define UNIMPLEMENTED_THEN_RETURN() OF_UNIMPLEMENTED()

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
