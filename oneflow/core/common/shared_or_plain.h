#ifndef ONEFLOW_CORE_COMMON_SHARED_OR_PLAIN_H_
#define ONEFLOW_CORE_COMMON_SHARED_OR_PLAIN_H_

#include <cstring>
#include <glog/logging.h>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename S, typename P>
class SharedOrPlain final {
 public:
  SharedOrPlain(P data) : shared_ptr_() { SetData(data); }
  SharedOrPlain(const SharedOrPlain& rhs);
  SharedOrPlain(const std::shared_ptr<S>& shared_ptr) : shared_ptr_(shared_ptr) {
    CHECK(!IsPlain());
  }
  ~SharedOrPlain();
  bool IsPlain() const;
  P plain_data() const;
  std::shared_ptr<S> shared_ptr() const;

  P operator*() const { return plain_data(); }

 private:
  struct PlainStruct final {
    uint64_t _ : 62, is_plain_data : 2;
    P data;
  };
  static_assert(sizeof(S*) == 8, "only 64-bit pointer supported");
  static_assert(sizeof(P) <= 8, "only plain data type supported");
  static_assert(sizeof(std::shared_ptr<S>) >= sizeof(PlainStruct),
                "unsupported shared_ptr implemenet");

  void SetData(P data);
  const PlainStruct* CastToPlainStruct() const;
  PlainStruct* MutCastToPlainStruct();

  std::shared_ptr<S> shared_ptr_;
};

template<typename S, typename P>
SharedOrPlain<S, P>::SharedOrPlain(const SharedOrPlain& rhs) {
  if (rhs.IsPlain()) {
    std::memcpy(this, &rhs, sizeof(*this));
  } else {
    shared_ptr_ = rhs.shared_ptr_;
  }
}

template<typename S, typename P>
const typename SharedOrPlain<S, P>::PlainStruct* SharedOrPlain<S, P>::CastToPlainStruct() const {
  const PlainStruct* __attribute__((__may_alias__)) ptr =
      reinterpret_cast<const PlainStruct*>(&shared_ptr_);
  return ptr;
}

template<typename S, typename P>
typename SharedOrPlain<S, P>::PlainStruct* SharedOrPlain<S, P>::MutCastToPlainStruct() {
  PlainStruct* __attribute__((__may_alias__)) ptr = reinterpret_cast<PlainStruct*>(&shared_ptr_);
  return ptr;
}

template<typename S, typename P>
void SharedOrPlain<S, P>::SetData(P data) {
  PlainStruct* const ptr = MutCastToPlainStruct();
  ptr->is_plain_data = 1;
  ptr->data = data;
}

template<typename S, typename P>
std::shared_ptr<S> SharedOrPlain<S, P>::shared_ptr() const {
  CHECK(!IsPlain());
  return shared_ptr_;
}

template<typename S, typename P>
P SharedOrPlain<S, P>::plain_data() const {
  const PlainStruct* const ptr = CastToPlainStruct();
  CHECK(ptr->is_plain_data);
  return ptr->data;
}

template<typename S, typename P>
bool SharedOrPlain<S, P>::IsPlain() const {
  const PlainStruct* const ptr = CastToPlainStruct();
  return ptr->is_plain_data;
}

template<typename S, typename P>
SharedOrPlain<S, P>::~SharedOrPlain() {
  if (IsPlain()) {
    std::shared_ptr<S> empty_ptr;
    std::memcpy(&shared_ptr_, &empty_ptr, sizeof(empty_ptr));
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_PLAIN_H_
