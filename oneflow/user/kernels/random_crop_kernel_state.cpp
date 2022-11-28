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
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/random_crop_kernel_state.h"

namespace oneflow {

std::shared_ptr<RandomCropKernelState> CreateRandomCropKernelState(
    user_op::KernelInitContext* ctx) {
  int32_t num_attempts = ctx->Attr<int32_t>("num_attempts");
  CHECK(num_attempts >= 1);
  const std::vector<float>& random_aspect_ratio =
      ctx->Attr<std::vector<float>>("random_aspect_ratio");
  CHECK(random_aspect_ratio.size() == 2 && 0 < random_aspect_ratio.at(0)
        && random_aspect_ratio.at(0) <= random_aspect_ratio.at(1));
  const std::vector<float>& random_area = ctx->Attr<std::vector<float>>("random_area");
  CHECK(random_area.size() == 2 && 0 < random_area.at(0) && random_area.at(0) <= random_area.at(1));
  const user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  return std::shared_ptr<RandomCropKernelState>(new RandomCropKernelState(
      out_tensor_desc->shape().elem_cnt(), CHECK_JUST(GetOpKernelRandomSeed(ctx)),
      {random_aspect_ratio.at(0), random_aspect_ratio.at(1)},
      {random_area.at(0), random_area.at(1)}, num_attempts));
}

}  // namespace oneflow
