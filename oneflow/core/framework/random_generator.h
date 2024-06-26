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

#include <mutex>

#include "oneflow/core/ep/include/random_generator.h"
#include "oneflow/core/framework/auto_random_generator.h"
#include "oneflow/core/framework/device.h"

#include "oneflow/core/ep/cpu/cpu_random_generator.h"
#include "oneflow/core/ep/cuda/cuda_random_generator.h"
#include "oneflow/core/common/hash_container.h"

namespace oneflow {

class NdSbp;

namespace one {

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation.
static constexpr uint64_t default_rng_seed_val = 67280421310721;

class Tensor;

class Generator final {
 public:
  explicit Generator(const std::shared_ptr<ep::RandomGenerator>& internal);

  ~Generator() = default;

  void set_current_seed(uint64_t seed);

  uint64_t current_seed() const;

  void add_children_generator(Symbol<ParallelDesc> placement, Symbol<NdSbp> nd_sbp,
                              const std::shared_ptr<Generator>& generator);
  const HashMap<std::pair<Symbol<ParallelDesc>, Symbol<NdSbp>>, std::shared_ptr<one::Generator>>&
  children_generators() const;

  // Reset current generator by a non-deterministic random seed, and returns it.
  uint64_t seed();

  Maybe<Symbol<Device>> device() const;

  Maybe<Tensor> GetState() const;
  Maybe<void> SetState(const std::shared_ptr<Tensor>& state);

  const std::shared_ptr<ep::RandomGenerator>& internal() const { return internal_; }

  template<typename T>
  Maybe<T> Get(int device_index = -1) const {
    if (auto* internal = dynamic_cast<AutoGenerator*>(internal_.get())) {
      return internal->GetOrCreate<T>(device_index);
    }
    auto internal = std::dynamic_pointer_cast<T>(internal_);
    CHECK_NOTNULL_OR_RETURN(internal);
    if (device_index != -1) {
      CHECK_EQ_OR_RETURN(device_index, internal->device_index())
          << "Invalid device index " << device_index << " since the generator's device index is "
          << internal->device_index();
    }
    return internal;
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<ep::RandomGenerator> internal_;
  // children generator for eager global mode
  HashMap<std::pair<Symbol<ParallelDesc>, Symbol<NdSbp>>,  // NOLINT
          std::shared_ptr<one::Generator>>                 // NOLINT
      children_generators_;                                // NOLINT
};

Maybe<Generator> MakeGenerator(const std::string& device, int device_index = -1);
Maybe<Generator> MakeGenerator(DeviceType device, int device_index = -1);

Maybe<Generator> MakeAutoGenerator();
Maybe<Generator> MakeCPUGenerator();
Maybe<Generator> MakeCUDAGenerator();

Maybe<Generator> DefaultAutoGenerator();
Maybe<Generator> DefaultCPUGenerator();
Maybe<Generator> DefaultCUDAGenerator(int device_index = -1);

Maybe<Generator> DefaultGenerator(const std::string& device, int device_index = -1);
Maybe<Generator> DefaultGenerator(DeviceType device, int device_index = -1);

Maybe<Generator> ManualSeed(uint64_t seed);

Maybe<void> ManualSeed(uint64_t seed, const std::string& device, int device_index = -1);
Maybe<void> ManualSeed(uint64_t seed, DeviceType device, int device_index = -1);

Maybe<void> ManualSeedAllCudaGenerator(uint64_t seed);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RANDOM_GENERATOR_H_
