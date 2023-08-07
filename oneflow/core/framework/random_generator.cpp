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

#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/framework/auto_random_generator.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow {
namespace one {

namespace {

uint64_t GetNonDeterministicRandom() {
  std::random_device rd;
  // limit to 53 bits to ensure unique representation in double
  auto s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  return s;
}

Maybe<void> CPUSynchronize() {
  if (Singleton<VirtualMachine>::Get() != nullptr) { return vm::CurrentRankSync(); }
  return Maybe<void>::Ok();
}

}  // namespace

Generator::Generator(const std::shared_ptr<ep::RandomGenerator>& internal) : internal_(internal) {}

uint64_t Generator::current_seed() const { return internal_->current_seed(); }

void Generator::set_current_seed(uint64_t seed) {
  CHECK_JUST(CPUSynchronize());
  internal_->set_current_seed(seed);
}

uint64_t Generator::seed() {
  uint64_t seed = GetNonDeterministicRandom();
  set_current_seed(seed);
  return seed;
}

Maybe<Symbol<Device>> Generator::device() const {
  return Device::New(internal_->device_type_name(), internal_->device_index());
}

Maybe<Tensor> Generator::GetState() const {
  JUST(CPUSynchronize());
  int64_t state_size = internal_->GetStateSize();
  std::vector<uint8_t> state_data(state_size);
  internal_->GetState(state_size, state_data.data());
  const auto& device = JUST(Device::New("cpu"));
  const auto& state = JUST(functional::Empty(Shape{state_size}, DType::UInt8(), device,
                                             /*requires_grad=*/false, /*pin_memory=*/false));
  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(eager_blob_object->mut_dptr(), state_data.data(), state_size);
  };
  JUST(SyncAccessTensorWithTimeOut(state, callback, "mut"));
  return state;
}

Maybe<void> Generator::SetState(const std::shared_ptr<Tensor>& state) {
  const auto& device = JUST(state->device());
  if (device->type() != "cpu") {
    return Error::RuntimeError() << "Generator state should be host tensor.";
  }
  if (state->dtype() != DType::UInt8()) {
    return Error::RuntimeError() << "Generator state should be dtype=flow.uint8";
  }
  size_t state_size = state->shape()->elem_cnt();
  std::vector<uint8_t> state_data(state_size);
  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(state_data.data(), eager_blob_object->dptr(), state_size);
  };
  JUST(SyncAccessTensorWithTimeOut(state, callback, "const"));
  JUST(CPUSynchronize());
  internal_->SetState(state_size, state_data.data());
  return Maybe<void>::Ok();
}

Maybe<Generator> DefaultGenerator(const std::string& device, int device_index) {
  static auto* default_auto_generator =
      dynamic_cast<AutoGenerator*>(JUST(DefaultAutoGenerator())->internal().get());
  if (device_index == -1) { device_index = (device == "cpu" ? 0 : GlobalProcessCtx::LocalRank()); }
  return std::make_shared<Generator>(
      JUST(default_auto_generator->GetOrCreate(device, device_index)));
}

Maybe<Generator> DefaultAutoGenerator() {
  // Skip destructing to avoid calling symbols in other dynamic libraries when the global object is
  // released.
  static auto default_auto_generator = std::make_shared<Generator>(std::shared_ptr<AutoGenerator>(
      new AutoGenerator(GetNonDeterministicRandom()), [](AutoGenerator*) {}));
  return default_auto_generator;
}

Maybe<Generator> DefaultCPUGenerator() {
  static auto default_cpu_generator = JUST(DefaultGenerator("cpu", 0));
  return default_cpu_generator;
}

Maybe<Generator> DefaultCUDAGenerator(int device_index) {
#ifdef WITH_CUDA
  static int device_count = GetCudaDeviceCount();
#else
  static int device_count = 0;
#endif  // WITH_CUDA
  static std::vector<std::once_flag> init_flags(device_count);
  static std::vector<std::shared_ptr<Generator>> default_cuda_generator(device_count);

  if (device_index == -1) { device_index = GlobalProcessCtx::LocalRank(); }
  CHECK_OR_RETURN(device_index >= 0 && device_index < device_count)
      << "Invalid device index " << device_index;
  std::call_once(init_flags[device_index], [&]() {
    default_cuda_generator[device_index] = CHECK_JUST(DefaultGenerator("cuda", device_index));
  });
  return default_cuda_generator.at(device_index);
}

Maybe<Generator> MakeAutoGenerator() {
  return std::make_shared<Generator>(std::make_shared<AutoGenerator>(default_rng_seed_val));
}

Maybe<Generator> MakeCPUGenerator() {
  static auto device_mgr =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(DeviceType::kCPU);
  return std::make_shared<Generator>(device_mgr->CreateRandomGenerator(default_rng_seed_val, 0));
}

Maybe<Generator> MakeCUDAGenerator(int device_index) {
  static auto device_mgr =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(DeviceType::kCUDA);
  if (device_index == -1) { device_index = GlobalProcessCtx::LocalRank(); }
  return std::make_shared<Generator>(
      device_mgr->CreateRandomGenerator(default_rng_seed_val, device_index));
}

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
  if (device == "auto") {
    return std::make_shared<Generator>(std::make_shared<AutoGenerator>(default_rng_seed_val));
  }
  auto device_type = ep::DeviceManagerRegistry::GetDeviceTypeByDeviceTypeName(device);
  if (device_type == DeviceType::kInvalidDevice) {
    return Error::RuntimeError() << "Expected one of " << PrintGeneratorAvailableDevices()
                                 << " device type at start of device string: " << device;
  }
  auto device_mgr = Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(device_type);
  if (device_index == -1) { device_index = (device == "cpu" ? 0 : GlobalProcessCtx::LocalRank()); }
  return std::make_shared<Generator>(
      device_mgr->CreateRandomGenerator(default_rng_seed_val, device_index));
}

Maybe<Generator> DefaultGenerator(DeviceType device, int device_index) {
  return DefaultGenerator(*JUST(DeviceTag4DeviceType(device)), device_index);
}

Maybe<Generator> MakeGenerator(DeviceType device, int device_index) {
  return MakeGenerator(*JUST(DeviceTag4DeviceType(device)), device_index);
}

Maybe<Generator> ManualSeed(uint64_t seed) {
  const auto& default_auto_generator = JUST(DefaultAutoGenerator());
  default_auto_generator->set_current_seed(seed);
  return default_auto_generator;
}

Maybe<void> ManualSeed(uint64_t seed, const std::string& device, int device_index) {
  JUST(DefaultGenerator(device, device_index))->set_current_seed(seed);
  return Maybe<void>::Ok();
}

Maybe<void> ManualSeed(uint64_t seed, DeviceType device, int device_index) {
  return ManualSeed(seed, *JUST(DeviceTag4DeviceType(device)), device_index);
}

}  // namespace one
}  // namespace oneflow
