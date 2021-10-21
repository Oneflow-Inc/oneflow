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
#include <utility>
#include "glog/logging.h"
#include "oneflow/core/common/math_util.h"

namespace oneflow {

int64_t Gcd(int64_t m, int64_t n) {
  if (m < n) { std::swap(m, n); }
  if (n == 0) { return m; }
  CHECK_GT(m, 0);
  CHECK_GT(n, 0);
  return Gcd(n, m % n);
}

int64_t Lcm(int64_t m, int64_t n) { return m * n / Gcd(m, n); }

}  // namespace oneflow
