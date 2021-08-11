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

#include <memory>
#include <type_traits>

#include <glog/logging.h>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename StructT, typename ScalarT>
class SharedOrScalar final {
 public:
  static_assert(IsScalarType<ScalarT>::value, "ScalarT should be scalar type.");

  SharedOrScalar(const ScalarT& scalar_value) : scalar_value_(scalar_value), is_scalar_(true) {}

  SharedOrScalar(const std::shared_ptr<StructT>& shared_ptr) : is_scalar_(false) {
    new (MutableSharedStorage()) Shared(shared_ptr);
  }

  SharedOrScalar(const SharedOrScalar& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (MutableSharedStorage()) Shared(*(rhs.GetSharedStorage()));
    }
  }

  ~SharedOrScalar() {
    if (is_scalar_) {
      scalar_value_.~ScalarT();
    } else {
      (*MutableSharedStorage()).~Shared();
    }
  }

  bool IsScalar() const { return is_scalar_; }
  const ScalarT& scalar_value() const {
    CHECK(is_scalar_);
    return scalar_value_;
  }

  const std::shared_ptr<StructT>& shared_ptr() const {
    CHECK(!is_scalar_);
    return GetSharedStorage()->data;
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

  Shared* MutableSharedStorage() { return reinterpret_cast<Shared*>(&shared_value_); }

  const Shared* GetSharedStorage() const { return reinterpret_cast<const Shared*>(&shared_value_); }

  template<typename T, typename... Ts>
  using AlignedUnion = typename std::aligned_union<1, T, Ts...>::type;

  union {
    ScalarT scalar_value_;
    AlignedUnion<Shared> shared_value_;
  };
  bool is_scalar_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
