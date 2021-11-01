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
#ifndef ONEFLOW_CORE_VECTORIZED_CPU_CAPBILITY_H_
#define ONEFLOW_CORE_VECTORIZED_CPU_CAPBILITY_H_

namespace oneflow {

enum class CPUCapability { DEFAULT = 0, AVX2 = 1, AVX512 = 2, NUM_OPTIONS };

CPUCapability compute_cpu_capability();

}  // namespace oneflow

#endif
