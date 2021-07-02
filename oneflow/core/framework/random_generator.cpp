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

uint64_t getNonDeterministicRandom() {
  std::random_device rd;
  // limit to 53 bits to ensure unique representation in double
  auto s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  return s;
}

/*static*/ Maybe<Generator> Generator::New(const std::string& device, uint64_t seed) {
  std::shared_ptr<Generator> generator(new Generator);
  JUST(generator->Init(device, seed));
  return generator;
}

Maybe<void> Generator::Init(const std::string& device, uint64_t seed) {
  if (device == "cpu") {
    gen_impl_ = std::make_shared<DeviceGeneratorImpl<DeviceType::kCPU>>(seed);
  }
#ifdef WITH_CUDA
  else if (device == "cuda") {
    gen_impl_ = std::make_shared<DeviceGeneratorImpl<DeviceType::kGPU>>(seed);
  }
#endif  // WITH_CUDA
  else if (device == "auto") {
    gen_impl_ = std::make_shared<AutoGeneratorImpl>(seed);
  } else {
    UNIMPLEMENTED_THEN_RETURN() << " device unimplemented, device name: " << device;
  }
  return Maybe<void>::Ok();
}

uint64_t Generator::seed() {
  uint64_t seed = getNonDeterministicRandom();
  set_current_seed(seed);
  return seed;
}

void ManualSeed(uint64_t seed) {
#ifdef WITH_CUDA
  const auto& cuda_gen = GetDefaultDeviceGenerator<DeviceType::kGPU>();
  cuda_gen->set_current_seed(seed);
#endif  // WITH_CUDA
  const auto& cpu_gen = GetDefaultDeviceGenerator<DeviceType::kCPU>();
  cpu_gen->set_current_seed(seed);
  const auto& auto_gen = GetDefaultAutoGenerator();
  auto_gen->set_current_seed(seed);
}

std::shared_ptr<AutoGeneratorImpl> CreateAutoGenerator(uint64_t seed) {
  return std::make_shared<AutoGeneratorImpl>(seed);
}

template<DeviceType device_type>
std::shared_ptr<DeviceGeneratorImpl<device_type>> CreateDeviceGenerator(uint64_t seed) {
  return std::make_shared<DeviceGeneratorImpl<device_type>>(seed);
}

const std::shared_ptr<AutoGeneratorImpl>& GetDefaultAutoGenerator() {
  static auto generator = CreateAutoGenerator(getNonDeterministicRandom());
  return generator;
}

template<DeviceType device_type>
const std::shared_ptr<DeviceGeneratorImpl<device_type>>& GetDefaultDeviceGenerator() {
  static auto generator = CreateDeviceGenerator<device_type>(getNonDeterministicRandom());
  return generator;
}

template<DeviceType device_type>
Maybe<DeviceGeneratorImpl<device_type>> TryGetDeviceGenerator(
    const std::shared_ptr<GeneratorImpl>& generator) {
  if (auto auto_gen = std::dynamic_pointer_cast<AutoGeneratorImpl>(generator)) {
    return auto_gen->template GetDeviceGenerator<device_type>();
  }
  auto device_gen = std::dynamic_pointer_cast<DeviceGeneratorImpl<device_type>>(generator);
  CHECK_NOTNULL_OR_RETURN(device_gen);
  return device_gen;
}

}  // namespace one
}  // namespace oneflow
