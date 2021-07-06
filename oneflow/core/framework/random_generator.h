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
#ifndef ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_

#include "oneflow/core/framework/random_generator_impl.h"

namespace oneflow {
namespace one {

Maybe<void> ManualSeed(uint64_t seed);

class Generator final {
 public:
  // The default seed is selected to be a large number
  // with good distribution of 0s and 1s in bit representation.
  static constexpr uint64_t default_rng_seed_val = 67280421310721;

 public:
  explicit Generator(const std::shared_ptr<AutoGeneratorImpl>& impl);
  explicit Generator(const std::shared_ptr<GeneratorImpl>& impl);

  ~Generator() = default;

  void set_current_seed(uint64_t seed);

  uint64_t current_seed() const;

  // Reset current seed by default seed, and returns it.
  uint64_t seed();

  const std::shared_ptr<AutoGeneratorImpl>& impl() const { return impl_; }

  template<DeviceType device_type>
  Maybe<GeneratorImpl> Get() const {
    return impl_->GetOrCreateDeviceGenerator<device_type>();
  }

 private:
  std::shared_ptr<AutoGeneratorImpl> impl_;
};

Maybe<Generator> GetDefaultAutoGenerator();
Maybe<Generator> MakeAutoGenerator(uint64_t seed);

template<DeviceType device_type>
Maybe<Generator> GetDefaultDeviceGenerator(int device_index = -1);

template<DeviceType device_type>
Maybe<Generator> MakeDeviceGenerator(uint64_t seed);

Maybe<Generator> MakeGenerator(const std::string& device);
Maybe<Generator> MakeGenerator(const std::string& device, uint64_t seed);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
