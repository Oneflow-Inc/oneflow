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
#include "oneflow/user/kernels/random_mask_generator.h"

namespace oneflow {

void RandomMaskGenerator<DeviceType::kCPU>::Generate(ep::Stream* stream, const int64_t n,
                                                     const float rate, bool* mask) {
  CHECK_GE(n, 0);
  std::uniform_real_distribution<float> random_distribution(GetZeroVal<float>(),
                                                            GetOneVal<float>());
  for (int64_t i = 0; i < n; ++i) { mask[i] = random_distribution(generator_->engine()) > rate; }
}

template class RandomMaskGenerator<DeviceType::kCPU>;

}  // namespace oneflow
