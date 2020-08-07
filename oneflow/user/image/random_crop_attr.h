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
#ifndef ONEFLOW_USER_IMAGE_RANDOM_CROP_ATTR_H_
#define ONEFLOW_USER_IMAGE_RANDOM_CROP_ATTR_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/image/random_crop_generator.h"

namespace oneflow {

class RandCropGens final : public user_op::OpKernelState {
 public:
  explicit RandCropGens(int32_t size) : gens_(size) {}
  ~RandCropGens() = default;

  RandomCropGenerator* Get(int32_t idx) { return gens_.at(idx).get(); }

  void New(int32_t idx, AspectRatioRange aspect_ratio_range, AreaRange area_range, int64_t seed,
           int32_t num_attempts) {
    CHECK_LT(idx, gens_.size());
    gens_.at(idx).reset(
        new RandomCropGenerator(aspect_ratio_range, area_range, seed, num_attempts));
  }

 private:
  std::vector<std::shared_ptr<RandomCropGenerator>> gens_;
};

std::shared_ptr<RandCropGens> CreateRandomCropState(user_op::KernelInitContext* ctx);

}  // namespace oneflow

#endif  // ONEFLOW_USER_IMAGE_RANDOM_CROP_ATTR_H_
