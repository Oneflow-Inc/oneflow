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
#include <iostream>
#include <string.h>
#include "cpuinfo.h"
#include "oneflow/core/vectorized/cpu_capbility.h"

namespace oneflow {

CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
#ifdef WITH_AVX
    if (strcmp(envar, "avx512") == 0) {
      std::cout << "ENV  avx512 " << std::endl;
      return CPUCapability::AVX512;
    }
    if (strcmp(envar, "avx2") == 0) {
      std::cout << "ENV  avx2 " << std::endl;
      return CPUCapability::AVX2;
    }
#endif
    if (strcmp(envar, "default") == 0) { return CPUCapability::DEFAULT; }
  }

  if (cpuinfo_initialize()) {

#ifdef WITH_AVX
    if (cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq()
        && cpuinfo_has_x86_fma3()) {
      std::cout << "CPUCapability  avx512 " << std::endl;
      return CPUCapability::AVX512;
    }
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      std::cout << "CPUCapability  avx2 " << std::endl;
      return CPUCapability::AVX2;
    }
#endif
  }

  std::cout << "CPUCapability  DEFAULT " << std::endl;
  return CPUCapability::DEFAULT;
}

}  // namespace oneflow