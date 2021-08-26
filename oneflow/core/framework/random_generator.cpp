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

#include <mutex>
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

Maybe<void> ManualSeed(uint64_t seed) {
  JUST(DefaultAutoGenerator())->set_current_seed(seed);
  return Maybe<void>::Ok();
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
  static std::vector<std::shared_ptr<Generator>> default_cuda_generator;
  static std::once_flag init_flags;
  static int device_count = 0;
  std::call_once(init_flags, [&]() {
    device_count = detail::GetCudaDeviceCount();
    default_cuda_generator.resize(device_count);
    for (int i = 0; i < device_count; ++i) {
      default_cuda_generator[i] = std::make_shared<Generator>(
          CHECK_JUST(CHECK_JUST(DefaultAutoGenerator())->Get<CUDAGeneratorImpl>(i)));
    }
  });
  if (device_index == -1) { OF_CUDA_CHECK(cudaGetDevice(&device_index)); }
  CHECK_OR_RETURN(device_index >= 0 && device_index < device_count)
      << "Invalid device index " << device_index;
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
  if (device_index == -1) { OF_CUDA_CHECK(cudaGetDevice(&device_index)); }
  CHECK_OR_RETURN(device_index >= 0 && device_index < detail::GetCudaDeviceCount())
      << "Invalid device index " << device_index;
  return std::make_shared<Generator>(
      std::make_shared<CUDAGeneratorImpl>(default_rng_seed_val, device_index));
}
#endif  // WITH_CUDA

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
    UNIMPLEMENTED_THEN_RETURN() << "Invalid device " << device
                                << " for making generator, please make sure the device is one of "
                                   "\"cpu\", \"cuda\" and \"auto\".";
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
    UNIMPLEMENTED_THEN_RETURN() << "Invalid device " << device
                                << " for making generator, please make sure the device is one of "
                                   "\"cpu\", \"cuda\" and \"auto\".";
  }
}

}  // namespace one
}  // namespace oneflow
