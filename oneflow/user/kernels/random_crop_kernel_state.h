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
#ifndef ONEFLOW_USER_KERNELS_RANDOM_CROP_KERNEL_STATE_H_
#define ONEFLOW_USER_KERNELS_RANDOM_CROP_KERNEL_STATE_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/image/random_crop_generator.h"

namespace oneflow {

class RandomCropKernelState final : public user_op::OpKernelState {
 public:
  explicit RandomCropKernelState(int32_t size, int64_t seed, AspectRatioRange aspect_ratio_range,
                                 AreaRange area_range, int32_t num_attempts)
      : gens_(size) {
    std::seed_seq seq{seed};
    std::vector<int> seeds(size);
    seq.generate(seeds.begin(), seeds.end());
    for (int32_t i = 0; i < size; ++i) {
      gens_.at(i).reset(
          new RandomCropGenerator(aspect_ratio_range, area_range, seeds.at(i), num_attempts));
    }
  }
  ~RandomCropKernelState() = default;

  RandomCropGenerator* GetGenerator(int32_t idx) { return gens_.at(idx).get(); }

 private:
  std::vector<std::shared_ptr<RandomCropGenerator>> gens_;
};

std::shared_ptr<RandomCropKernelState> CreateRandomCropKernelState(user_op::KernelInitContext* ctx);

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_RANDOM_CROP_KERNEL_STATE_H_
