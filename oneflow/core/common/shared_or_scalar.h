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
#include <memory>
#include <type_traits>

#include <glog/logging.h>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T, typename... Ts>
struct AlignedCharArrayUnion {
  using AlignedUnion = typename std::aligned_union<1, T, Ts...>::type;
  alignas(alignof(AlignedUnion)) char buffer[sizeof(AlignedUnion)];
};

template<typename StructT, typename ScalarT>
class SharedOrScalar final {
 public:
  SharedOrScalar(const ScalarT& scalar_value) : scalar_value_(scalar_value), is_scalar_(true) {}

  SharedOrScalar(const std::shared_ptr<StructT>& shared_ptr) : is_scalar_(false) {
    new (getSharedStorage()) Shared(shared_ptr);
  }

  SharedOrScalar(const SharedOrScalar& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (getSharedStorage()) Shared(*(rhs.getSharedStorage()));
    }
  }

  ~SharedOrScalar() {
    if (is_scalar_) {
      scalar_value_.~ScalarT();
    } else {
      (*getSharedStorage()).~Shared();
    }
  }

  bool IsScalar() const { return is_scalar_; }
  const ScalarT& scalar_value() const {
    CHECK(is_scalar_);
    return scalar_value_;
  }

  const std::shared_ptr<StructT>& shared_ptr() const {
    CHECK(!is_scalar_);
    return getSharedStorage()->data;
  }

  const ScalarT& operator*() const { return scalar_value(); }

 private:
  struct Shared {
    Shared() = default;
    Shared(const std::shared_ptr<StructT>& v) : data(v) {}
    Shared(std::shared_ptr<StructT>&& v) : data(std::move(v)) {}
    virtual ~Shared() = default;

    std::shared_ptr<StructT> data;
  };

  Shared* getSharedStorage() { return reinterpret_cast<Shared*>(&shared_value_); }

  const Shared* getSharedStorage() const { return reinterpret_cast<const Shared*>(&shared_value_); }

  union {
    ScalarT scalar_value_;
    AlignedCharArrayUnion<Shared> shared_value_;
  };
  bool is_scalar_;

  static_assert(sizeof(StructT*) == 8, "only 64-bit pointer supported");
  static_assert(sizeof(ScalarT) <= 8, "only scalar data type supported");
  static_assert(sizeof(std::shared_ptr<StructT>) >= sizeof(ScalarT),
                "unsupported shared_ptr implemenet");
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
