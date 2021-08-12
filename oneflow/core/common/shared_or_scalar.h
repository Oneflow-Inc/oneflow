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

  using Shared = std::shared_ptr<StructT>;

  SharedOrScalar(const ScalarT& scalar_value) : is_scalar_(true), scalar_value_(scalar_value) {}

  SharedOrScalar(const std::shared_ptr<StructT>& shared_ptr)
      : is_scalar_(false), shared_value_(shared_ptr) {}

  SharedOrScalar(std::shared_ptr<StructT>&& shared_ptr)
      : is_scalar_(false), shared_value_(std::move(shared_ptr)) {}

  SharedOrScalar(const SharedOrScalar& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (&shared_value_) Shared(rhs.shared_value_);
    }
  }

  SharedOrScalar(SharedOrScalar&& rhs) : is_scalar_(rhs.is_scalar_) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      new (&shared_value_) Shared(std::move(rhs.shared_value_));
    }
  }

  SharedOrScalar& operator=(const SharedOrScalar& rhs) {
    if (rhs.is_scalar_) {
      scalar_value_ = rhs.scalar_value_;
    } else {
      if (is_scalar_) {
        new (&shared_value_) Shared(rhs.shared_value_);
      } else {
        shared_value_ = rhs.shared_value_;
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
        new (&shared_value_) Shared(std::move(rhs.shared_value_));
      } else {
        shared_value_ = std::move(rhs.shared_value_);
      }
    }
    is_scalar_ = rhs.is_scalar_;
    return *this;
  }

  ~SharedOrScalar() {
    if (is_scalar_) {
      scalar_value_.~ScalarT();
    } else {
      shared_value_.~Shared();
    }
  }

  bool IsScalar() const { return is_scalar_; }
  const ScalarT& scalar_value() const {
    CHECK(is_scalar_);
    return scalar_value_;
  }

  const std::shared_ptr<StructT>& shared_ptr() const {
    CHECK(!is_scalar_);
    return shared_value_;
  }

  const ScalarT& operator*() const { return scalar_value(); }

 private:
  bool is_scalar_;
  union {
    ScalarT scalar_value_;
    Shared shared_value_;
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHARED_OR_SCALAR_H_
