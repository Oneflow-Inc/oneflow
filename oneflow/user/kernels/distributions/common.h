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
#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS_COMMON_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS_COMMON_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/random_generator.h"

namespace oneflow {

class DistributionKernelState : public user_op::OpKernelState {
 public:
  explicit DistributionKernelState(const std::shared_ptr<one::Generator>& generator)
      : generator_(generator) {}

  const std::shared_ptr<one::Generator>& generator() const { return generator_; }

 private:
  std::shared_ptr<one::Generator> generator_;
};

// FIXME: refine warning message
#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  CHECK(var >= min && var <= max) << name << " is out of bounds for " << dtype;

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype)                                          \
  if (var < -(1LL << digits) || var > (1LL << digits)) {                                      \
    LOG(WARNING) << name << " is out of bounds [-(2^" << digits << "), 2^" << digits << "]. " \
                 << "Due to precision limitations " << dtype                                  \
                 << " can support discrete uniform distribution only within this range. "     \
                 << "This warning will become an error in later version release.";            \
  }

template<typename scalar_t>
void check_from_to_in_range(int64_t from, int64_t to_inc) {
  if (IsFloating<scalar_t>::value) {
    const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
    CHECK_OUT_OF_BOUNDS(from, "from", min, max, GetDataType<scalar_t>::value);
    CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, GetDataType<scalar_t>::value);

    constexpr auto digits = std::numeric_limits<scalar_t>::digits;
    WARN_OUT_OF_BOUNDS(from, "from", digits, GetDataType<scalar_t>::value);
    WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, GetDataType<scalar_t>::value);
  } else if (IsIntegral<scalar_t>::value || IsUnsignedIntegral<scalar_t>::value) {
    const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
    CHECK_OUT_OF_BOUNDS(from, "from", min, max, GetDataType<scalar_t>::value);
    CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, GetDataType<scalar_t>::value);
  } else {
    UNIMPLEMENTED()
        << "check_random_bounds handles only integral, floating-point and boolean types";
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_KERNEL_H_
