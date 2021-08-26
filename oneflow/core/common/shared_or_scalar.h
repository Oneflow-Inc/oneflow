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

#include <glog/logging.h>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename StructT, typename ScalarT>
class SharedOrScalar final {
 public:
  static_assert(IsScalarType<ScalarT>::value, "ScalarT should be scalar type.");

  using Shared = std::shared_ptr<StructT>;

  SharedOrScalar(const ScalarT& scalar_value) : is_scalar_(true), scalar_value_(scalar_value) {}

  SharedOrScalar(const std::shared_ptr<StructT>& shared_ptr) : is_scalar_(false) {
    new (&shared_mem_) Shared(shared_ptr);
  }

  SharedOrScalar(std::shared_ptr<StructT>&& shared_ptr) : is_scalar_(false) {
    new (&shared_mem_) Shared(std::move(shared_ptr));
  }

  SharedOrScalar(const SharedOrScalar& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (&shared_mem_) Shared(rhs.GetShared());
    }
  }

  SharedOrScalar(SharedOrScalar&& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (&shared_mem_) Shared(std::move(*rhs.MutableShared()));
    }
  }

  SharedOrScalar& operator=(const SharedOrScalar& rhs) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      if (is_scalar_) {
        scalar_value_.~ScalarT();
        new (&shared_mem_) Shared(rhs.GetShared());
      } else {
        *MutableShared() = rhs.GetShared();
      }
    }
    is_scalar_ = rhs.is_scalar_;
    return *this;
  }

  SharedOrScalar& operator=(SharedOrScalar&& rhs) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      if (is_scalar_) {
        scalar_value_.~ScalarT();
        new (&shared_mem_) Shared(std::move(*rhs.MutableShared()));
      } else {
        *MutableShared() = std::move(*rhs.MutableShared());
      }
    }
    is_scalar_ = rhs.is_scalar_;
    return *this;
  }

  ~SharedOrScalar() {
    if (is_scalar_) {
      scalar_value_.~ScalarT();
    } else {
      GetShared().~Shared();
    }
  }

  bool IsScalar() const { return is_scalar_; }
  const ScalarT& scalar_value() const {
    CHECK(is_scalar_);
    return scalar_value_;
  }

  const std::shared_ptr<StructT>& shared_ptr() const {
    CHECK(!is_scalar_);
    return GetShared();
  }

  const ScalarT& operator*() const { return scalar_value(); }

 private:
  bool is_scalar_;
  union {
    ScalarT scalar_value_;

    //  to avoid error(a non-POD class definition is not allowed inside of a statement expression)
    //  in nvcc while using with JUST macro (this type is used in Maybe)
    alignas(Shared) char shared_mem_[sizeof(Shared)];
  };

  const Shared& GetShared() const {
    const auto* __attribute__((__may_alias__)) shared =
        reinterpret_cast<const Shared*>(&shared_mem_);
    return *shared;
  }

  Shared* MutableShared() {
    auto* __attribute__((__may_alias__)) shared = reinterpret_cast<Shared*>(&shared_mem_);
    return shared;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
