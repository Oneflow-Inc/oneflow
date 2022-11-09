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
#include "oneflow/core/framework/to_string.h"

#include <mutex>
#include "oneflow/core/control/global_process_ctx.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

Maybe<Generator> ManualSeed(uint64_t seed) {
  const auto& default_auto_generator = JUST(DefaultAutoGenerator());
  default_auto_generator->set_current_seed(seed);
  return default_auto_generator;
}

Maybe<void> ManualSeed(uint64_t seed, const std::string& device, int device_index) {
  if (device == "cpu") {
    JUST(DefaultCPUGenerator())->set_current_seed(seed);
  }
#ifdef WITH_CUDA
  else if (device == "cuda") {
    JUST(DefaultCUDAGenerator(device_index))->set_current_seed(seed);
  }
#endif  // WITH_CUDA
  else if (device == "auto") {
    JUST(DefaultAutoGenerator())->set_current_seed(seed);
  } else {
    return Error::RuntimeError() << "Invalid device " << device
                                 << " for making generator, please make sure the device is one of "
                                 << PrintGeneratorAvailableDevices();
  }
  return Maybe<void>::Ok();
}

Maybe<void> ManualSeed(uint64_t seed, DeviceType device, int device_index) {
  return ManualSeed(seed, *JUST(DeviceTag4DeviceType(device)), device_index);
}

namespace detail {

uint64_t GetNonDeterministicRandom() {
  std::random_device rd;
  // limit to 53 bits to ensure unique representation in double
  auto s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  return s;
}

}  // namespace detail

Generator::Generator(const std::shared_ptr<GeneratorImpl>& impl) : impl_(impl) {}

uint64_t Generator::current_seed() const { return impl_->current_seed(); }

void Generator::set_current_seed(uint64_t seed) { impl_->set_current_seed(seed); }

uint64_t Generator::seed() {
  uint64_t seed = detail::GetNonDeterministicRandom();
  set_current_seed(seed);
  return seed;
}

Maybe<Generator> DefaultAutoGenerator() {
  static auto default_auto_generator = std::make_shared<Generator>(
      std::make_shared<AutoGeneratorImpl>(detail::GetNonDeterministicRandom()));
  return default_auto_generator;
}

Maybe<Generator> DefaultCPUGenerator() {
  static auto default_cpu_generator =
      std::make_shared<Generator>(JUST(JUST(DefaultAutoGenerator())->Get<CPUGeneratorImpl>(0)));
  return default_cpu_generator;
}

#ifdef WITH_CUDA
Maybe<Generator> DefaultCUDAGenerator(int device_index) {
  static int device_count = GetCudaDeviceCount();
  static std::vector<std::once_flag> init_flags(device_count);
  static std::vector<std::shared_ptr<Generator>> default_cuda_generator(device_count);

  if (device_index == -1) { device_index = GetCudaDeviceIndex(); }
  CHECK_OR_RETURN(device_index >= 0 && device_index < device_count)
      << "Invalid device index " << device_index;
  std::call_once(init_flags[device_index], [&]() {
    default_cuda_generator[device_index] = std::make_shared<Generator>(
        CHECK_JUST(CHECK_JUST(DefaultAutoGenerator())->Get<CUDAGeneratorImpl>(device_index)));
  });
  return default_cuda_generator.at(device_index);
}
#endif  // WITH_CUDA

Maybe<Generator> MakeAutoGenerator() {
  return std::make_shared<Generator>(std::make_shared<AutoGeneratorImpl>(default_rng_seed_val));
}

Maybe<Generator> MakeCPUGenerator() {
  return std::make_shared<Generator>(std::make_shared<CPUGeneratorImpl>(default_rng_seed_val));
}

#ifdef WITH_CUDA
Maybe<Generator> MakeCUDAGenerator(int device_index) {
  if (device_index == -1) { device_index = GetCudaDeviceIndex(); }
  CHECK_OR_RETURN(device_index >= 0 && device_index < GetCudaDeviceCount())
      << "Invalid device index " << device_index;
  return std::make_shared<Generator>(
      std::make_shared<CUDAGeneratorImpl>(default_rng_seed_val, device_index));
}
#endif  // WITH_CUDA

Maybe<void> ManualSeedAllCudaGenerator(uint64_t seed) {
#ifdef WITH_CUDA
  static int device_count = GetCudaDeviceCount();
  FOR_RANGE(int, device_id, 0, device_count) {
    const auto& cuda_gen = JUST(DefaultCUDAGenerator(device_id));
    cuda_gen->set_current_seed(seed);
  }
#endif  // WITH_CUDA
  return Maybe<void>::Ok();
}

Maybe<Generator> MakeGenerator(const std::string& device, int device_index) {
  if (device == "cpu") {
    return MakeCPUGenerator();
  }
#ifdef WITH_CUDA
  else if (device == "cuda") {
    return MakeCUDAGenerator(device_index);
  }
#endif  // WITH_CUDA
  else if (device == "auto") {
    return MakeAutoGenerator();
  } else {
    return Error::RuntimeError() << "Invalid device " << device
                                 << " for making generator, please make sure the device is one of "
                                 << PrintGeneratorAvailableDevices();
  }
}

Maybe<Generator> DefaultGenerator(const std::string& device, int device_index) {
  if (device == "cpu") {
    return DefaultCPUGenerator();
  }
#ifdef WITH_CUDA
  else if (device == "cuda") {
    return DefaultCUDAGenerator(device_index);
  }
#endif  // WITH_CUDA
  else if (device == "auto") {
    return DefaultAutoGenerator();
  } else {
    return Error::RuntimeError() << "Invalid device " << device
                                 << " for making generator, please make sure the device is one of "
                                 << PrintGeneratorAvailableDevices();
  }
}

Maybe<Generator> DefaultGenerator(DeviceType device, int device_index) {
  return DefaultGenerator(*JUST(DeviceTag4DeviceType(device)), device_index);
}

Maybe<Generator> MakeGenerator(DeviceType device, int device_index) {
  return MakeGenerator(*JUST(DeviceTag4DeviceType(device)), device_index);
}

}  // namespace one
}  // namespace oneflow
