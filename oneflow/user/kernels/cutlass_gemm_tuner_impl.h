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

#ifdef WITH_CUTLASS

#ifndef ONEFLOW_USER_KERNELS_CUTLASS_GEMM_TUNER_IMPL_H_
#define ONEFLOW_USER_KERNELS_CUTLASS_GEMM_TUNER_IMPL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>
#include <cutlass/library/singleton.h>

namespace oneflow {

template<typename Configuration, typename Arguments>
class CutlassGemmTunerImpl {
 public:
  const cutlass::library::Operation* Find(ep::CudaStream* stream,
                                          cutlass::library::GemmFunctionalKey functional_key,
                                          const Configuration& configuraion,
                                          const Arguments& arguments, void* workspace,
                                          size_t workspace_size);

  const cutlass::library::Operation* Get(const std::string& name, ep::CudaStream* stream,
                                         cutlass::library::GemmFunctionalKey functional_key,
                                         const Configuration& configuraion,
                                         const Arguments& arguments, void* workspace,
                                         size_t workspace_size);
};

template<typename Configuration, typename Arguments>
CutlassGemmTunerImpl<Configuration, Arguments>* GetCutlassGemmTunerImpl();

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUTLASS_GEMM_TUNER_IMPL_H_

#endif  // WITH_CUTLASS
