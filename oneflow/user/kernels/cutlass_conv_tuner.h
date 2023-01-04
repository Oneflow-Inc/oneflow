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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include <cutlass/library/handle.h>
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>

namespace oneflow {

class CutlassConvTuner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CutlassConvTuner);
  ~CutlassConvTuner() = default;

  const cutlass::library::Operation* FindConv2dOperation(
      ep::CudaStream* stream, cutlass::library::ConvFunctionalKey functional_key,
      const cutlass::library::Conv2dConfiguration& configuraion,
      const cutlass::library::ConvArguments& arguments, void* workspace,
      size_t workspace_size) const;

  const cutlass::library::Operation* GetConv2dOperation(
      const std::string& name, ep::CudaStream* stream,
      cutlass::library::ConvFunctionalKey functional_key,
      const cutlass::library::Conv2dConfiguration& configuraion,
      const cutlass::library::ConvArguments& arguments, void* workspace,
      size_t workspace_size) const;

  static const CutlassConvTuner& Get();

 private:
  CutlassConvTuner();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace oneflow

#endif  // WITH_CUTLASS
#endif  // ONEFLOW_USER_KERNELS_CUTLASS_CONV_TUNER_H_
