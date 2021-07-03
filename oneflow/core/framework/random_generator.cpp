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
#include "oneflow/core/framework/random_generator.h"

namespace oneflow {
namespace one {

/**
 * [Copy from pytorch]
 * Gets a non deterministic random number number from either the
 * /dev/urandom or the current time. For CUDA, gets random from
 * std::random_device and adds a transformation on it.
 *
 * FIXME: The behavior in this function is from legacy code
 * (THRandom_seed/THCRandom_seed) and is probably not the right thing to do,
 * even though our tests pass. Figure out if tests get perturbed
 * - when the same algorithm is used for all backends. Note that the current
 * behavior is different for CPU, CUDA and Windows CPU.
 * - when using C++11 std objects, such as std::random_device
 * - when constructing a 64 bit seed properly, rather than static casting
 *   a 32 bit number to 64 bit.
 */
uint64_t getNonDeterministicRandom() {
  std::random_device rd;
  // limit to 53 bits to ensure unique representation in double
  auto s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  return s;
}

void manual_seed(uint64_t seed) {
#ifdef WITH_CUDA
  const auto& cuda_gen = GetDefaultDeviceGenerator<DeviceType::kGPU>();
  cuda_gen->set_seed(seed);
#endif  // WITH_CUDA
  const auto& cpu_gen = GetDefaultDeviceGenerator<DeviceType::kCPU>();
  cpu_gen->set_seed(seed);
  const auto& auto_gen = GetDefaultAutoGenerator();
  auto_gen->set_seed(seed);
}

std::shared_ptr<AutoGeneratorImpl> CreateAutoGenerator(uint64_t seed) {
  return std::make_shared<AutoGeneratorImpl>(seed);
}

const std::shared_ptr<AutoGeneratorImpl>& GetDefaultAutoGenerator() {
  static auto generator = CreateAutoGenerator(getNonDeterministicRandom());
  return generator;
}

template<DeviceType device_type>
const Maybe<DeviceGeneratorImpl<device_type>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImpl>& generator) {
  CHECK_NOTNULL_OR_RETURN(generator);
  if (generator->device_type() == "auto") {
    const auto& auto_gen = std::dynamic_pointer_cast<AutoGeneratorImpl>(generator);
    CHECK_NOTNULL_OR_RETURN(auto_gen);
    return auto_gen->template GetDeviceGenerator<device_type>();
  }
  const auto& device_gen = std::dynamic_pointer_cast<DeviceGeneratorImpl<device_type>>(generator);
  CHECK_NOTNULL_OR_RETURN(device_gen);
  return device_gen;
}

template const Maybe<DeviceGeneratorImpl<DeviceType::kCPU>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImpl>& generator);
template const Maybe<DeviceGeneratorImpl<DeviceType::kGPU>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImpl>& generator);

template<DeviceType device_type>
const Maybe<DeviceGeneratorImpl<device_type>> TryGetDeviceGenerator(
    const std::shared_ptr<Generator>& generator) {
  CHECK_NOTNULL_OR_RETURN(generator);
  return TryGetDeviceGenerator<device_type>(generator->get_impl());
}

template const Maybe<DeviceGeneratorImpl<DeviceType::kCPU>> TryGetDeviceGenerator(
    const std::shared_ptr<Generator>& generator);
template const Maybe<DeviceGeneratorImpl<DeviceType::kGPU>> TryGetDeviceGenerator(
    const std::shared_ptr<Generator>& generator);

}  // namespace one
}  // namespace oneflow
