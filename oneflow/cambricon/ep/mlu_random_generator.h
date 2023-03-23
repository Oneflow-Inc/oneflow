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
#ifndef ONEFLOW_CAMBRICON_EP_MLU_RANDOM_GENERATOR_H_
#define ONEFLOW_CAMBRICON_EP_MLU_RANDOM_GENERATOR_H_

#include <mutex>

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/ep/include/random_generator.h"

namespace oneflow {
namespace ep {

class MLUGenerator : public RandomGenerator {
 public:
  explicit MLUGenerator(uint64_t seed, int device_index);
  virtual ~MLUGenerator();

  uint64_t current_seed() const override { return seed_; }
  void set_current_seed(uint64_t seed) override;

  std::string device_type_name() const override { return "mlu"; }
  int64_t device_index() const override { return device_index_; }

  size_t GetStateSize() const override;
  void GetState(size_t state_size, void* state) const override;
  void SetState(size_t state_size, const void* state) override;

  bool need_update_state() const { return need_update_state_; }
  void* state() const { return state_; }

  cnnlRandGenerator_t cnnl_rng() const { return cnnl_rng_; }

  void update_state(cnnlHandle_t handle);

 public:
  mutable std::mutex mutex_;

 private:
  uint64_t seed_;
  int64_t device_index_;
  size_t state_size_;
  bool need_update_state_;
  void* state_;
  cnnlRandGenerator_t cnnl_rng_;
};

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_EP_MLU_RANDOM_GENERATOR_H_
