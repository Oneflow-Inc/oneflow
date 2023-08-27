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
#ifndef ONEFLOW_USER_KERNELS_CUTLASS_CONV_TUNER_H_
#define ONEFLOW_USER_KERNELS_CUTLASS_CONV_TUNER_H_

#ifdef WITH_CUTLASS

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/cutlass_conv_tuner_impl.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>

namespace oneflow {

class CutlassConvTuner {
 public:
  CutlassConvTuner() = default;

  template<typename Configuration, typename Arguments>
  const cutlass::library::Operation* FindConv2dOperation(
      ep::CudaStream* stream, const cutlass::library::ConvFunctionalKey& functional_key,
      const Configuration& configuraion, const Arguments& arguments, void* workspace,
      size_t workspace_size) {
    return GetCutlassConvTunerImpl<Configuration, Arguments>()->Find(
        stream, functional_key, configuraion, arguments, workspace, workspace_size);
  }

  template<typename Configuration, typename Arguments>
  const cutlass::library::Operation* GetConv2dOperation(
      const std::string& name, ep::CudaStream* stream,
      const cutlass::library::ConvFunctionalKey& functional_key, const Configuration& configuraion,
      const Arguments& arguments, void* workspace, size_t workspace_size) {
    return GetCutlassConvTunerImpl<Configuration, Arguments>()->Get(
        name, stream, functional_key, configuraion, arguments, workspace, workspace_size);
  }
};

}  // namespace oneflow

#endif  // WITH_CUTLASS

#endif  // ONEFLOW_USER_KERNELS_CUTLASS_CONV_TUNER_H_
