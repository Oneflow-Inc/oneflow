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

#include "algorithm_util.h"

namespace oneflow {
namespace auto_parallel {

// Inverse function of order
// The reason why we need the InvOrder, a.k.a id2order, instead of id2value is to eliminate
// equality. For example, we have v[0] < v[1] = v[2] < v[3] We do not know v[1] is before or after
// v[2] with comp(v[1], v[2]). But if we transfer it to order order[0] < order[1] < order[2] <
// order[3] We know the strict order.
void InverseOrder(const std::vector<int32_t>& order, std::vector<int32_t>& InvOrder) {
  InvOrder.resize(order.size());
  for (int32_t i = 0; i < order.size(); i++) { InvOrder[order[i]] = i; }
}

}  // namespace auto_parallel
}  // namespace oneflow
