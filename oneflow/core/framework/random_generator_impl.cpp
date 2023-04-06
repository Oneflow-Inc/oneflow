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
#include "oneflow/core/framework/random_generator_impl.h"

#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

namespace {

Maybe<void> CPUSynchronize() {
  if (Singleton<VirtualMachine>::Get() != nullptr) { return vm::CurrentRankSync(); }
  return Maybe<void>::Ok();
}

}  // namespace

struct CPUGeneratorState {
  static constexpr int64_t state_size = std::mt19937::state_size;  // 624
  int64_t states[state_size] = {};
  int64_t seed = 0;
};
constexpr int64_t CPUGeneratorState::state_size;

void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
  CHECK_JUST(CPUSynchronize());
  seed_ = seed;
  engine_.seed(seed_);
  torch_engine_ = pytorch_mt19937_engine(seed);
}

Maybe<Tensor> CPUGeneratorImpl::GetState() const {
  JUST(CPUSynchronize());
  CPUGeneratorState state;
  const auto& device = JUST(Device::New("cpu"));
  const auto& tensor_state = JUST(functional::Empty(Shape{sizeof(state)}, DType::UInt8(), device,
                                                    /*requires_grad=*/false, /*pin_memory=*/false));

  std::stringstream ss;
  ss << engine_;
  std::vector<std::string> splits;
  Split(ss.str(), " ", [&](std::string&& s) { splits.emplace_back(s); });
  // The last element in `splits` indicates state size, not state.
  if (splits.size() != CPUGeneratorState::state_size + 1) {
    return Error::RuntimeError() << "std::mt19937 state size should be "
                                 << CPUGeneratorState::state_size << ", but got "
                                 << splits.size() - 1;
  }
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) {
    state.states[i] = std::atoll(JUST(VectorAt(splits, i)).data());
  }
  state.seed = current_seed();

  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(eager_blob_object->mut_dptr(), &state, sizeof(state));
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "mut"));
  return tensor_state;
}

Maybe<void> CPUGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  JUST(CPUSynchronize());
  const auto& device = JUST(tensor_state->device());
  if (device->type() != "cpu") {
    return Error::RuntimeError() << "Generator state should be host tensor.";
  }
  if (tensor_state->dtype() != DType::UInt8()) {
    return Error::RuntimeError() << "Generator state should be dtype=flow.uint8";
  }
  CPUGeneratorState state;
  if (tensor_state->shape()->elem_cnt() != sizeof(state)) {
    return Error::RuntimeError() << "Tensor state size is not match for CPU generator. It needs "
                                 << sizeof(state) << ", but got "
                                 << tensor_state->shape()->elem_cnt();
  }
  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(reinterpret_cast<void*>(&state), eager_blob_object->dptr(), sizeof(state));
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "const"));

  // set_current_seed(state.seed);
  seed_ = state.seed;

  std::stringstream ss;
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) { ss << state.states[i] << " "; }
  ss << CPUGeneratorState::state_size;
  ss >> engine_;
  return Maybe<void>::Ok();
}

#ifdef WITH_CUDA
namespace {

int GetThreadNum(const cudaDeviceProp& prop) {
  switch (prop.major) {
    case 3:  // Kepler
      return 2 * 192;
    case 5:  // Maxwell
      return 2 * 128;
    case 6:  // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) {
        return 2 * 128;
      } else {
        return 2 * 64;
      }
    case 7:  // Volta and Turing
      return 2 * 64;
    default: return 2 * 64;
  }
}

}  // namespace

CUDAGeneratorImpl::CUDAGeneratorImpl(uint64_t seed, int device_index)
    : DeviceGeneratorImpl(seed, DeviceType::kCUDA, device_index), philox_offset_per_thread_(0) {
  cudaDeviceProp prop;  // NOLINT
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
  max_block_num_ = prop.multiProcessorCount;
  max_thread_num_ = GetThreadNum(prop);
}

void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  CHECK_JUST(CPUSynchronize());
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

std::tuple<uint64_t, dim3, dim3> CUDAGeneratorImpl::CalcExecutionPolicy(int64_t total_elements,
                                                                        ep::CudaStream* stream) {
  // NOTE(Liang Depeng): the implementation is modified from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/DistributionTemplates.h

  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = 256;  // block_size_bound
  // number of randoms given by distributions like curand_uniform4, curand_uniform2_double
  // used in calculating philox offset.
  const uint32_t curand4_engine_calls = 4;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = stream->device_properties().maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(stream->device_properties().multiProcessorCount) * blocks_per_sm,
      grid.x);
  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  uint64_t counter_offset =
      ((numel - 1) / (block_size * grid.x * unroll) + 1) * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// NOTE(Liang Depeng): The implementation of ` CUDAGeneratorImpl::get_philox_offset` is modified
// from
//      https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L269
//      in order to make distribution related cuda kernels to have the same output as pytorch
//      when setting the same seed.
uint64_t CUDAGeneratorImpl::get_philox_offset(uint64_t increment) {
  std::lock_guard<std::mutex> lock(mutex_);
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  CHECK_EQ(this->philox_offset_per_thread_ % 4, 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return offset;
}

Maybe<Tensor> CUDAGeneratorImpl::GetState() const {
  JUST(CPUSynchronize());
  // NOTE: The RNG state comprises the seed, and an offset used for Philox.
  // The following line is just here for aligning Pytorch and it is also no
  // practical effect in Pytorch just for backward compatibility reason.
  // For more details pls refer to:
  // https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L152
  static const size_t states_size = 200 * sizeof(4120);
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;
  const auto& device = JUST(Device::New("cpu"));
  const auto& tensor_state = JUST(functional::Empty(Shape{total_size}, DType::UInt8(), device,
                                                    /*requires_grad=*/false, /*pin_memory=*/false));

  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memset(static_cast<uint8_t*>(eager_blob_object->mut_dptr()), -1, states_size);
    memcpy(static_cast<uint8_t*>(eager_blob_object->mut_dptr()) + states_size, &seed_, seed_size);
    memcpy(static_cast<uint8_t*>(eager_blob_object->mut_dptr()) + states_size + seed_size,
           &philox_offset_per_thread_, offset_size);
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "mut"));
  return tensor_state;
}

Maybe<void> CUDAGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  static const size_t states_size = 200 * sizeof(4120);  // this line is just here for BC reason
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;

  const auto& device = JUST(tensor_state->device());
  if (device->type() != "cpu") {
    return Error::RuntimeError() << "Generator state should be host tensor.";
  }
  if (tensor_state->dtype() != DType::UInt8()) {
    return Error::RuntimeError() << "Generator state should be dtype=flow.uint8";
  }
  if (tensor_state->shape()->elem_cnt() != total_size) {
    return Error::RuntimeError() << "Tensor state size is not match for CUDA generator. It needs "
                                 << total_size << ", but got " << tensor_state->shape()->elem_cnt();
  }

  JUST(CPUSynchronize());
  const auto& callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    const uint8_t* data = static_cast<const uint8_t*>(eager_blob_object->dptr());
    // Do not use set_current_seed() since synchronization will lead to deadlock.
    seed_ = *((uint64_t*)(data + states_size));
    philox_offset_per_thread_ = *((uint64_t*)(data + states_size + seed_size));
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "const"));
  return Maybe<void>::Ok();
}
#endif  // WITH_CUDA

void AutoGeneratorImpl::set_current_seed(uint64_t seed) {
  CHECK_JUST(CPUSynchronize());
  std::lock_guard<std::mutex> lock(mutex_);
  seed_ = seed;
  for (const auto& it : generators_) {
    if (unlikely(pthread_fork::IsForkedSubProcess() && it.first.device_type == kCUDA)) { continue; }
    it.second->set_current_seed(seed);
  }
}

struct AutoGeneratorState {
  uint64_t seed = 0;
  int64_t num = 0;
  int64_t device_tag_length = 0;
  int64_t state_length = 0;
  // std::vector<int64_t> state_sizes[num];
  // std::vector<uint8_t> device_tags[device_tag_length];
  // std::vector<uint8_t> states[state_sizes[0] + state_sizes[1] + ... + state_sizes[num - 1]]
};

Maybe<Tensor> AutoGeneratorImpl::GetState() const {
  JUST(CPUSynchronize());
  std::lock_guard<std::mutex> lock(mutex_);

  AutoGeneratorState state;
  state.seed = current_seed();
  state.num = generators_.size();

  state.state_length = 0;
  std::vector<std::shared_ptr<Tensor>> tensor_states;
  std::vector<int64_t> state_sizes;
  state_sizes.reserve(generators_.size());
  for (auto it = generators_.begin(); it != generators_.end(); ++it) {
    const auto& tensor_state = JUST(it->second->GetState());
    tensor_states.emplace_back(tensor_state);
    state_sizes.emplace_back(tensor_state->shape()->elem_cnt());
    state.state_length += state_sizes.back();
  }

  std::stringstream ss;
  auto it = generators_.begin();
  if (it != generators_.end()) {
    ss << JUST(it->second->device())->ToString();
    ++it;
  }
  for (; it != generators_.end(); ++it) { ss << "," << JUST(it->second->device())->ToString(); }
  std::string device_tags = ss.str();
  state.device_tag_length = device_tags.size();

  int64_t total_size =
      sizeof(state) + state.num * sizeof(int64_t) + state.device_tag_length + state.state_length;
  std::vector<uint8_t> buffer(total_size);
  {
    uint8_t* data = buffer.data();
    memcpy(data, &state, sizeof(state));
    data += sizeof(state);
    memcpy(data, state_sizes.data(), state.num * sizeof(int64_t));
    data += state.num * sizeof(int64_t);
    memcpy(data, device_tags.data(), state.device_tag_length);
    data += state.device_tag_length;
    for (int i = 0; i < tensor_states.size(); ++i) {
      const auto& tensor = tensor_states[i];
      const auto& callback = [&data, &state_sizes, i](
                                 ep::Stream*,
                                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
        memcpy(data, eager_blob_object->dptr(), state_sizes.at(i));
      };
      JUST(SyncAccessTensorWithTimeOut(tensor, callback, "const"));
      data += JUST(VectorAt(state_sizes, i));
    }
  }
  const auto& device = JUST(Device::New("cpu"));
  const auto& tensor_state = JUST(functional::Empty(Shape{total_size}, DType::UInt8(), device,
                                                    /*requires_grad=*/false, /*pin_memory=*/false));
  const auto& callback = [&buffer, &total_size](
                             ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(eager_blob_object->mut_dptr(), buffer.data(), total_size);
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "mut"));
  return tensor_state;
}

Maybe<void> AutoGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  const auto& device = JUST(tensor_state->device());
  if (device->type() != "cpu") {
    return Error::RuntimeError() << "Generator state should be host tensor.";
  }
  if (tensor_state->dtype() != DType::UInt8()) {
    return Error::RuntimeError() << "Generator state should be dtype=flow.uint8";
  }
  AutoGeneratorState state;
  int64_t total_size = tensor_state->shape()->elem_cnt();
  std::vector<uint8_t> buffer(total_size);
  const auto& callback = [&buffer, &total_size](
                             ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    memcpy(buffer.data(), eager_blob_object->dptr(), total_size);
  };
  JUST(SyncAccessTensorWithTimeOut(tensor_state, callback, "const"));

  const uint8_t* data = buffer.data();
  memcpy(reinterpret_cast<void*>(&state), data, sizeof(state));
  if (total_size
      != sizeof(state) + state.num * sizeof(int64_t) + state.device_tag_length
             + state.state_length) {
    return Error::RuntimeError() << "Invalid auto generator state, size is not match.";
  }
  data += sizeof(state);
  std::vector<int64_t> state_sizes(state.num);
  memcpy(state_sizes.data(), data, state.num * sizeof(int64_t));
  data += state.num * sizeof(int64_t);
  std::string device_tags;
  device_tags.resize(state.device_tag_length);
  memcpy(const_cast<char*>(device_tags.data()), data, state.device_tag_length);
  data += state.device_tag_length;
  std::vector<std::shared_ptr<Tensor>> tensor_states(state.num);
  for (int i = 0; i < state.num; ++i) {
    int64_t state_size = JUST(VectorAt(state_sizes, i));
    tensor_states[i] = JUST(functional::Empty(Shape{state_size}, DType::UInt8(), device,
                                              /*requires_grad=*/false, /*pin_memory=*/false));
    const auto& callback = [&data, &state_size](
                               ep::Stream*,
                               const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
      memcpy(eager_blob_object->mut_dptr(), data, state_size);
    };
    JUST(SyncAccessTensorWithTimeOut(tensor_states[i], callback, "mut"));
    data += state_size;
  }

  // set current seed.
  set_current_seed(state.seed);

  std::vector<std::string> splits;
  Split(device_tags, ",", [&](std::string&& s) { splits.emplace_back(s); });
  if (splits.size() != state.num) {
    return Error::RuntimeError() << "Invalid auto generator state. The number of state is "
                                 << state.num << ", but device tags number is " << splits.size();
  }
  JUST(CPUSynchronize());
  std::lock_guard<std::mutex> lock(mutex_);

  for (int i = 0; i < splits.size(); ++i) {
    const auto& device = JUST(Device::ParseAndNew(splits[i]));
    detail::DeviceKey device_key;
    device_key.device_type = JUST(DeviceType4DeviceTag(device->type()));
    device_key.device_index = device->device_id();
    auto it = generators_.find(device_key);
    if (it == generators_.end()) {
      it = generators_
               .emplace(device_key, JUST(detail::MakeGeneratorImpl(seed_, device_key.device_type,
                                                                   device_key.device_index)))
               .first;
    }
    JUST(it->second->SetState(JUST(VectorAt(tensor_states, i))));
  }
  return Maybe<void>::Ok();
}

namespace detail {

bool operator==(const DeviceKey& k1, const DeviceKey& k2) {
  return k1.device_type == k2.device_type && k1.device_index == k2.device_index;
}

size_t DeviceKeyHash::operator()(const DeviceKey& key) const {
  return (static_cast<uint64_t>(key.device_type) << 10) + key.device_index;
}

template<>
DeviceKey MakeDeviceKey<CPUGeneratorImpl>(int device_index) {
  if (device_index < 0) { device_index = 0; }
  DeviceKey device_key;
  device_key.device_type = DeviceType::kCPU;
  device_key.device_index = device_index;
  return device_key;
}

template<>
Maybe<CPUGeneratorImpl> MakeGeneratorImpl<CPUGeneratorImpl>(uint64_t seed, int device_index) {
  return std::make_shared<CPUGeneratorImpl>(seed);
}

#ifdef WITH_CUDA

template<>
DeviceKey MakeDeviceKey<CUDAGeneratorImpl>(int device_index) {
  if (device_index == -1) { device_index = GetCudaDeviceIndex(); }
  DeviceKey device_key;
  device_key.device_type = DeviceType::kCUDA;
  device_key.device_index = device_index;
  return device_key;
}

template<>
Maybe<CUDAGeneratorImpl> MakeGeneratorImpl<CUDAGeneratorImpl>(uint64_t seed, int device_index) {
  CHECK_OR_RETURN(device_index >= 0 && device_index < GetCudaDeviceCount())
      << "Invalid device index " << device_index;
  return std::make_shared<CUDAGeneratorImpl>(seed, device_index);
}
#endif  // WITH_CUDA

Maybe<GeneratorImpl> MakeGeneratorImpl(uint64_t seed, DeviceType device_type, int device_index) {
  std::shared_ptr<GeneratorImpl> impl;
  switch (device_type) {
    case kCPU: {
      impl = JUST(MakeGeneratorImpl<CPUGeneratorImpl>(seed, device_index));
      break;
    }
#ifdef WITH_CUDA
    case kCUDA: {
      impl = JUST(MakeGeneratorImpl<CUDAGeneratorImpl>(seed, device_index));
      break;
    }
#endif  // WITH_CUDA
    default:
      return Error::RuntimeError() << "Can not make generator for device type " << device_type;
  }
  return impl;
}

}  // namespace detail

}  // namespace one
}  // namespace oneflow
