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
#include <omp.h>
#include <iostream>
#include <sys/sysinfo.h>
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/cpu/parallel.h"

namespace oneflow {
namespace ep {
namespace primitive {

#if WITH_OMP_THREADING_RUNTIME
#define BUZY_NUM 4

int Parallel::_computing_cores;

void Parallel::set_computing_cores() {
  auto envar = std::getenv("ONEFLOW_CPU_CORES");
  if (envar) {
    _computing_cores = std::stoi(envar);
    return;
  }

  int cpu_core = get_nprocs();
  _computing_cores = (cpu_core / GlobalProcessCtx::NumOfProcessPerNode()) - BUZY_NUM;
  if (_computing_cores < 1) { _computing_cores = 1; }
}

#endif
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
