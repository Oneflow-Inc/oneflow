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
#ifndef ONEFLOW_CORE_FRAMEWORK_AUTO_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_FRAMEWORK_AUTO_RANDOM_GENERATOR_H_

#include <mutex>
#include <unordered_map>
#include <vector>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/random_generator.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace one {

class AutoGenerator : public ep::RandomGenerator {
 public:
  AutoGenerator(uint64_t seed) : seed_(seed) {}
  virtual ~AutoGenerator() = default;

  uint64_t current_seed() const override { return seed_; }
  void set_current_seed(uint64_t seed) override;

  std::string device_type_name() const override { return "auto"; }
  int64_t device_index() const override { return 0; }

  size_t GetStateSize() const override;
  void GetState(size_t state_size, void* state) const override;
  void SetState(size_t state_size, const void* state) override;

  Maybe<ep::RandomGenerator> GetOrCreate(const std::string& device, int device_index);

  template<typename T>
  Maybe<T> GetOrCreate(int device_index) {
    return std::dynamic_pointer_cast<T>(
        JUST(GetOrCreate(ep::GetRandomGeneratorDeviceTypeName<T>(), device_index)));
  }

 private:
  mutable std::mutex mutex_;
  uint64_t seed_;
  std::unordered_map<Symbol<Device>, std::shared_ptr<ep::RandomGenerator>> generators_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_AUTO_RANDOM_GENERATOR_H_
