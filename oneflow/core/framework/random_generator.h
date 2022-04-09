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

class Tensor;

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation.
static constexpr uint64_t default_rng_seed_val = 67280421310721;

class Generator final {
 public:
  explicit Generator(const std::shared_ptr<GeneratorImpl>& impl);

  ~Generator() = default;

  void set_current_seed(uint64_t seed);

  uint64_t current_seed() const;

  // Reset current generator by a non-deterministic random seed, and returns it.
  uint64_t seed();

  Maybe<Symbol<Device>> device() const { return impl_->device(); }

  Maybe<Tensor> GetState() const { return impl_->GetState(); }
  Maybe<void> SetState(const std::shared_ptr<Tensor>& state) { return impl_->SetState(state); }

  const std::shared_ptr<GeneratorImpl>& impl() const { return impl_; }

  template<typename T>
  Maybe<T> Get(int device_index = -1) const {
    if (auto* impl = dynamic_cast<AutoGeneratorImpl*>(impl_.get())) {
      return impl->GetOrCreate<T>(device_index);
    }
    auto impl = std::dynamic_pointer_cast<T>(impl_);
    CHECK_NOTNULL_OR_RETURN(impl);
    if (device_index != -1) {
      CHECK_EQ_OR_RETURN(device_index, impl->device_index())
          << "Invalid device index " << device_index << " since the generator's device index is "
          << impl->device_index();
    }
    return impl;
  }

 private:
  std::shared_ptr<GeneratorImpl> impl_;
};

Maybe<Generator> ManualSeed(uint64_t seed);

Maybe<void> ManualSeed(uint64_t seed, const std::string& device, int device_index = -1);
Maybe<void> ManualSeed(uint64_t seed, DeviceType device, int device_index = -1);

Maybe<Generator> DefaultGenerator(const std::string& device, int device_index = -1);
Maybe<Generator> DefaultGenerator(DeviceType device, int device_index = -1);

Maybe<Generator> MakeGenerator(const std::string& device, int device_index = -1);
Maybe<Generator> MakeGenerator(DeviceType device, int device_index = -1);

Maybe<Generator> DefaultAutoGenerator();
Maybe<Generator> MakeAutoGenerator();

Maybe<Generator> DefaultCPUGenerator();
Maybe<Generator> MakeCPUGenerator();

#ifdef WITH_CUDA
Maybe<Generator> DefaultCUDAGenerator(int device_index = -1);
Maybe<Generator> MakeCUDAGenerator();
#endif  // WITH_CUDA
Maybe<void> ManualSeedAllCudaGenerator(uint64_t seed);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
