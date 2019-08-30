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
  const Error& error() const { return *data_or_error_.template Get<const Error>(); }

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
  Maybe() : MaybeBase<void>(std::shared_ptr<void>()) {}
  Maybe(const Error& error) : MaybeBase<void>(std::make_shared<const Error>(error)) {}
  Maybe(const std::shared_ptr<const Error>& error) : MaybeBase<void>(error) {}
  Maybe(const Error* error) : MaybeBase<void>(std::shared_ptr<const Error>(error)) {}
  Maybe(const Maybe<void>&) = default;
  ~Maybe() override = default;

  static Maybe<void> Ok() { return Maybe<void>(); }
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
      return maybe;                                            \
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

#endif  // ONEFLOW_CORE_COMMON_MAYBE_H_
