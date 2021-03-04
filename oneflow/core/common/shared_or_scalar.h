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
#ifndef ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
#define ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_

#include <cstring>
#include <glog/logging.h>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename StructT, typename ScalarT>
class SharedOrScalar final {
 public:
  SharedOrScalar(ScalarT scalar_value) : shared_ptr_() { SetScalar(scalar_value); }
  SharedOrScalar(const SharedOrScalar& rhs) { *this = rhs; }
  SharedOrScalar(const std::shared_ptr<StructT>& shared_ptr) : shared_ptr_(shared_ptr) {
    CHECK(!IsScalar());
  }
  ~SharedOrScalar();

  SharedOrScalar& operator=(const SharedOrScalar& rhs);

  bool IsScalar() const;
  ScalarT scalar_value() const;
  std::shared_ptr<StructT> shared_ptr() const;

  ScalarT operator*() const { return scalar_value(); }

 private:
  struct ScalarStruct final {
    uint64_t _ : 62, is_scalar_value : 2;
    ScalarT scalar_value;
  };
  static_assert(sizeof(StructT*) == 8, "only 64-bit pointer supported");
  static_assert(sizeof(ScalarT) <= 8, "only scalar data type supported");
  static_assert(sizeof(std::shared_ptr<StructT>) >= sizeof(ScalarStruct),
                "unsupported shared_ptr implemenet");

  void SetScalar(ScalarT scalar_value);
  const ScalarStruct* CastToScalarStruct() const;
  ScalarStruct* MutCastToScalarStruct();

  std::shared_ptr<StructT> shared_ptr_;
};

template<typename StructT, typename ScalarT>
SharedOrScalar<StructT, ScalarT>& SharedOrScalar<StructT, ScalarT>::operator=(
    const SharedOrScalar<StructT, ScalarT>& rhs) {
  if (rhs.IsScalar()) {
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    std::memcpy(this, &rhs, sizeof(*this));
  } else {
    shared_ptr_ = rhs.shared_ptr_;
  }
  return *this;
}

template<typename StructT, typename ScalarT>
const typename SharedOrScalar<StructT, ScalarT>::ScalarStruct*
SharedOrScalar<StructT, ScalarT>::CastToScalarStruct() const {
  const ScalarStruct* __attribute__((__may_alias__)) ptr =
      reinterpret_cast<const ScalarStruct*>(&shared_ptr_);
  return ptr;
}

template<typename StructT, typename ScalarT>
typename SharedOrScalar<StructT, ScalarT>::ScalarStruct*
SharedOrScalar<StructT, ScalarT>::MutCastToScalarStruct() {
  ScalarStruct* __attribute__((__may_alias__)) ptr = reinterpret_cast<ScalarStruct*>(&shared_ptr_);
  return ptr;
}

template<typename StructT, typename ScalarT>
void SharedOrScalar<StructT, ScalarT>::SetScalar(ScalarT scalar_value) {
  ScalarStruct* const ptr = MutCastToScalarStruct();
  ptr->is_scalar_value = 1;
  ptr->scalar_value = scalar_value;
}

template<typename StructT, typename ScalarT>
std::shared_ptr<StructT> SharedOrScalar<StructT, ScalarT>::shared_ptr() const {
  CHECK(!IsScalar());
  return shared_ptr_;
}

template<typename StructT, typename ScalarT>
ScalarT SharedOrScalar<StructT, ScalarT>::scalar_value() const {
  const ScalarStruct* const ptr = CastToScalarStruct();
  CHECK(ptr->is_scalar_value);
  return ptr->scalar_value;
}

template<typename StructT, typename ScalarT>
bool SharedOrScalar<StructT, ScalarT>::IsScalar() const {
  const ScalarStruct* const ptr = CastToScalarStruct();
  return ptr->is_scalar_value;
}

template<typename StructT, typename ScalarT>
SharedOrScalar<StructT, ScalarT>::~SharedOrScalar() {
  if (IsScalar()) {
    std::shared_ptr<StructT> empty_ptr;
    std::memcpy(&shared_ptr_, &empty_ptr, sizeof(empty_ptr));
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
